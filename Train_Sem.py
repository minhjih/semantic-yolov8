import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
import glob
import os
import cv2
import numpy as np
import gc

# [핵심] 메모리 파편화 방지 설정 (가장 윗부분에 유지)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    'model_name': 'yolov8n.pt',
    'data_path': '/fast/coco',   
    'batch_size': 8,             # [유지] OOM 방지를 위해 8로 설정
    'lr': 1e-4,
    'epochs': 100,
    'compression_ratio': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'img_size': 640,
    'lambda_p3': 1.0,
    'lambda_p4': 1.0,
    'lambda_p5': 1.0
}

# ==========================================
# 2. 데이터셋 (Simple COCO)
# ==========================================
class SimpleCocoDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
        if not self.image_paths:
            print(f"!!! 경고: {root_dir} 에서 이미지를 찾지 못했습니다. Dummy 데이터를 사용합니다.")
            self.image_paths = ["dummy"] * 100
        else:
            print(f">>> 데이터 로드 완료: {len(self.image_paths)}장")
        self.img_size = CONFIG['img_size']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if path == "dummy":
            img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            try:
                img = cv2.imread(path)
                if img is None: raise ValueError("Img None")
                img = cv2.resize(img, (self.img_size, self.img_size))
            except Exception:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) 
        return img

# ==========================================
# 3. Student 모델 (Semantic Autoencoder)
# ==========================================
class SemanticAutoencoder(nn.Module):
    def __init__(self, compression_ratio=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256 // compression_ratio, kernel_size=1) 
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 // compression_ratio, 256, kernel_size=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# ==========================================
# 4. Feature Extractor
# ==========================================
class MultiScaleFeatureExtractor:
    def __init__(self, model):
        self.hooks = []
        self.features = {}
        self.target_layers = {4: 'P3', 6: 'P4', 9: 'P5'} 
        for layer_idx, name in self.target_layers.items():
            hook = model.model[layer_idx].register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.features[name] = output[0]
            else:
                self.features[name] = output
        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()
    def clear(self):
        self.features = {}

# ==========================================
# 5. 메인 학습 루프
# ==========================================
def main():
    # [시작 전 메모리 정리]
    gc.collect()
    torch.cuda.empty_cache()
    
    device = CONFIG['device']
    print(f">>> Device: {device}")
    
    os.makedirs('model', exist_ok=True)
    
    print(">>> Loading Teacher Model (YOLOv8n)...")
    yolo = YOLO(CONFIG['model_name'])
    teacher_model = yolo.model.to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    student = SemanticAutoencoder(compression_ratio=CONFIG['compression_ratio']).to(device)
    optimizer = optim.Adam(student.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    # [AMP] Scaler 초기화 (메모리 절약 필수)
    scaler = torch.amp.GradScaler('cuda')

    dataset = SimpleCocoDataset(CONFIG['data_path'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    extractor = MultiScaleFeatureExtractor(teacher_model)

    print(">>> Start Training with AMP & Memory Optimization...")

    for epoch in range(CONFIG['epochs']):
        total_epoch_loss = 0.0
        
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device) 

            optimizer.zero_grad()

            # -------------------------------------------------
            # [A] Teacher (Original) -> GT 생성
            # -------------------------------------------------
            extractor.clear()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # 원본 이미지의 최종 결과 (비교용으로 저장)
                    original_final_output = teacher_model(imgs)
            
            target_features = {k: v.detach() for k, v in extractor.features.items()}

            # -------------------------------------------------
            # [B] & [C] Student -> Teacher (Fake) -> Loss
            # -------------------------------------------------
            extractor.clear()
            
            with torch.amp.autocast('cuda'):
                # 1. Reconstruction
                recon_imgs = student(imgs)
                
                # 2. Perceptual Loss (Backprop through Teacher)
                recon_final_output = teacher_model(recon_imgs)
                
                student_features = {k: v for k, v in extractor.features.items()}

                # 3. Loss 계산
                loss_p3 = criterion(student_features['P3'], target_features['P3'])
                loss_p4 = criterion(student_features['P4'], target_features['P4'])
                loss_p5 = criterion(student_features['P5'], target_features['P5'])
                total_loss = CONFIG['lambda_p3'] * loss_p3 + CONFIG['lambda_p4'] * loss_p4 + CONFIG['lambda_p5'] * loss_p5

            # [AMP] Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += total_loss.item()

            # -------------------------------------------------
            # [D] 로그 및 비교 (50 Iteration 마다) - [수정됨]
            # -------------------------------------------------
            if (i + 1) % 50 == 0:
                print(f"[Epoch {epoch}][Iter {i}] Total Loss: {total_loss.item():.4f}")

                # 텐서 추출 (Tuple이나 List인 경우 처리)
                t1, t2 = None, None
                
                # Original Output
                if isinstance(original_final_output, (list, tuple)) and len(original_final_output) > 0:
                    t1 = original_final_output[0]
                elif isinstance(original_final_output, torch.Tensor):
                    t1 = original_final_output
                    
                # Recon Output
                if isinstance(recon_final_output, (list, tuple)) and len(recon_final_output) > 0:
                    t2 = recon_final_output[0]
                elif isinstance(recon_final_output, torch.Tensor):
                    t2 = recon_final_output
                
                if t1 is not None and t2 is not None:
                    # 비교 연산 시 그래디언트 필요 없음
                    with torch.no_grad():
                        # t2(Recon 결과)는 그래프에 연결되어 있으므로 detach() 필수
                        # -----------------------------------------------------------
                        # [핵심] 텐서 슬라이싱 (YOLOv8 output: [Batch, 84, Anchors])
                        # 0~3번 채널: bbox (cx, cy, w, h) -> 픽셀 단위
                        # 4~83번 채널: class scores -> 0~1 확률 단위
                        # -----------------------------------------------------------
                        
                        # [1] Box 좌표 차이 (평균 픽셀 오차)
                        box_diff = (t1[:, :4, :] - t2.detach()[:, :4, :]).abs().mean().item()
                        
                        # [2] Class 확률 차이 (평균 확률 오차)
                        cls_diff = (t1[:, 4:, :] - t2.detach()[:, 4:, :]).abs().mean().item()
                        
                    print(f"    >> [Hybrid Check] Box Diff: {box_diff:.2f} px  |  Class Diff: {cls_diff:.5f}")

            # [E] 모델 저장
            if (i + 1) % 1000 == 0:
                save_path = os.path.join('model', 'sem_temp.pth')
                torch.save(student.state_dict(), save_path)
                print(f"    >>> Model Updated: {save_path}")

        avg_loss = total_epoch_loss / len(dataloader)
        print(f">>> [Epoch {epoch} Done] Avg Loss: {avg_loss:.4f}")
        
        # Epoch 끝날 때마다 캐시 정리
        torch.cuda.empty_cache()

    extractor.remove()

if __name__ == "__main__":
    main()