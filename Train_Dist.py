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

# [핵심] 메모리 파편화 방지 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    'model_name': 'yolov8n.pt',
    'data_path': '/fast/coco',   
    'split_layer': 4,              # 앞에서부터 몇번째 layer를 기준으로 나눌지          
    'compression_ratio': 4,        # student model의 압축률
    'student_layer_num': 3,        # student model의 layer 수
    'batch_size': 8,               # [수정] 16 -> 8 (OOM 방지)
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'img_size': 640
}

# ==========================================
# 2. 데이터셋
# ==========================================
class SimpleCocoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
        if not self.image_paths:
            print(f"경고: {root_dir}에서 이미지를 찾을 수 없습니다. 더미 데이터를 생성합니다.")
            self.image_paths = ["dummy"] * 100
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
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                img = cv2.resize(img, (self.img_size, self.img_size))
            except Exception:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) 
        return img

# ==========================================
# 3. Student 모델
# ==========================================
class StudentCNN(nn.Module):
    def __init__(self, target_channels, num_layers=3, compression_ratio=4):
        super().__init__()
        self.target_c = target_channels
        self.hidden_c = int(target_channels / compression_ratio)
        base_channels = 64  

        # [A] Encoder
        encoder_layers = []
        in_c = 3 
        
        for i in range(num_layers - 1):
            stride = 2 if i < 3 else 1 
            encoder_layers.append(nn.Conv2d(in_c, base_channels, kernel_size=3, stride=stride, padding=1))
            encoder_layers.append(nn.BatchNorm2d(base_channels))
            encoder_layers.append(nn.ReLU())
            in_c = base_channels 

        encoder_layers.append(nn.Conv2d(in_c, self.hidden_c, kernel_size=1))
        self.encoder = nn.Sequential(*encoder_layers)

        # [B] Decoder
        decoder_layers = []
        current_c = self.hidden_c
        
        for i in range(num_layers - 1):
            stride = 2 if i < 3 else 1
            decoder_layers.append(nn.ConvTranspose2d(current_c, base_channels, kernel_size=4, stride=stride, padding=1))
            decoder_layers.append(nn.BatchNorm2d(base_channels))
            decoder_layers.append(nn.ReLU())
            current_c = base_channels

        decoder_layers.append(nn.Conv2d(current_c, target_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ==========================================
# 4. 메인 함수
# ==========================================
def main():
    # [메모리 정리]
    gc.collect()
    torch.cuda.empty_cache()

    device = CONFIG['device']
    os.makedirs('model', exist_ok=True)
    
    # 1. Teacher Model 준비
    print(">>> Loading Teacher Model (YOLOv8n)...")
    yolo_full = YOLO(CONFIG['model_name'])
    teacher_model = yolo_full.model.to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # 2. Teacher N번째 레이어 정보 파악
    split_idx = CONFIG['split_layer']
    dummy_input = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).to(device)
    
    temp_shape = []
    def get_shape_hook(module, input, output):
        temp_shape.append(output.shape)
    
    hook_handle = teacher_model.model[split_idx].register_forward_hook(get_shape_hook)
    teacher_model(dummy_input)
    hook_handle.remove()
    
    target_shape = temp_shape[0] 
    target_channels = target_shape[1]
    print(f">>> Target Shape: {target_shape}, Student Layers(L): {CONFIG['student_layer_num']}")

    # 3. Student Model & Optimizer
    student = StudentCNN(
        target_channels=target_channels, 
        num_layers=CONFIG['student_layer_num'], 
        compression_ratio=CONFIG['compression_ratio']
    ).to(device)
    
    optimizer = optim.Adam(student.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    # [AMP] Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # 4. 데이터 로더
    dataset = SimpleCocoDataset(CONFIG['data_path'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # 5. Hybrid Inference용 Hook 관련 변수
    global student_feature_buffer
    student_feature_buffer = None

    def replace_feature_hook(module, input, output):
        global student_feature_buffer
        if student_feature_buffer is not None:
            # 크기 보정 (혹시 모를 불일치 대비)
            if output.shape != student_feature_buffer.shape:
                student_feature_buffer = nn.functional.interpolate(
                    student_feature_buffer, size=(output.shape[2], output.shape[3]), mode='bilinear'
                )
            return student_feature_buffer
        return output

    # 6. 학습용 Hook (Feature 추출)
    feature_storage = []
    def get_feature_hook(module, input, output):
        feature_storage.append(output)

    print(">>> Start Training with AMP...")
    
    for epoch in range(5):
        total_epoch_loss = 0.0
        
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            
            # -----------------------------------------------------------
            # [A] Teacher Forward (GT Feature & Original Output)
            # -----------------------------------------------------------
            feature_storage.clear()
            handle = teacher_model.model[split_idx].register_forward_hook(get_feature_hook)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # 여기서 나온 original_output은 나중에 평가(비교)에 재사용
                    original_output = teacher_model(imgs)
            
            handle.remove()
            target_feature = feature_storage[0].detach()

            # -----------------------------------------------------------
            # [B] Student Forward & Loss
            # -----------------------------------------------------------
            with torch.amp.autocast('cuda'):
                student_feature = student(imgs)

                # 크기 보정
                if student_feature.shape != target_feature.shape:
                    student_feature = nn.functional.interpolate(
                        student_feature, size=(target_feature.shape[2], target_feature.shape[3]), mode='bilinear'
                    )

                loss = criterion(student_feature, target_feature)

            # [C] Backward (with Scaler)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += loss.item()

            # -----------------------------------------------------------
            # [D] 50번마다 평가 (Box vs Class 분리)
            # -----------------------------------------------------------
            if (i + 1) % 50 == 0:
                print(f"[Epoch {epoch}][Iter {i}] L2 Loss: {loss.item():.6f}")
                
                # Hybrid 실행을 위해 student feature 저장
                student_feature_buffer = student_feature.detach()
                
                # Hook 등록 후 Hybrid Forward
                hook_handle = teacher_model.model[split_idx].register_forward_hook(replace_feature_hook)
                
                with torch.no_grad():
                    # 비교용이므로 여기서도 no_grad
                    hybrid_output = teacher_model(imgs)
                
                hook_handle.remove()
                student_feature_buffer = None # 버퍼 비우기

                # [수정] 결과 텐서 추출 및 비교
                t1, t2 = None, None
                
                # Original Output 추출
                if isinstance(original_output, (list, tuple)) and len(original_output) > 0:
                    t1 = original_output[0]
                elif isinstance(original_output, torch.Tensor):
                    t1 = original_output
                    
                # Hybrid Output 추출
                if isinstance(hybrid_output, (list, tuple)) and len(hybrid_output) > 0:
                    t2 = hybrid_output[0]
                elif isinstance(hybrid_output, torch.Tensor):
                    t2 = hybrid_output
                
                if t1 is not None and t2 is not None:
                    # -------------------------------------------------------
                    # [핵심] Metric 분리 (Box: 0~3, Class: 4~)
                    # -------------------------------------------------------
                    # t2는 detach 필요 (이미 no_grad 블록이지만 명시적으로 안전하게)
                    
                    # Box Coordinates Difference (Pixel Scale)
                    box_diff = (t1[:, :4, :] - t2[:, :4, :]).abs().mean().item()
                    
                    # Class Probability Difference (0~1 Scale)
                    cls_diff = (t1[:, 4:, :] - t2[:, 4:, :]).abs().mean().item()
                    
                    print(f"    >> [Hybrid Check] Box Diff: {box_diff:.2f} px | Class Diff: {cls_diff:.5f}")
            
            # [E] 모델 저장
            if (i + 1) % 1000 == 0:
                save_path = os.path.join('model', 'dist_temp.pth')
                torch.save(student.state_dict(), save_path)
                print(f"    >>> Model Updated: {save_path} (Iter {i})")
        
        # Epoch 종료 시 정리
        avg_loss = total_epoch_loss / len(dataloader)
        print(f">>> [Epoch {epoch} Done] Avg Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()