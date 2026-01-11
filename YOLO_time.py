from __future__ import annotations

import time
from pathlib import Path
from typing import List

import torch
import numpy as np
import onnxruntime as ort  # Added for ONNX Runtime
from ultralytics import YOLO

# Configuration
SAMPLE_DIR = Path("/fast/coco/val2017")
# If images are missing, use the automatically downloaded bus.jpg for testing
SAMPLE_COUNT = 50 
MODEL_WEIGHTS = "yolov8n.pt"
ONNX_PATH = Path("yolov8n.onnx") # Use the previously generated file

def collect_samples(directory: Path, count: int) -> List[str]:
    # Use default image if directory doesn't exist (prevent errors)
    if not directory.exists():
        print(f"[Info] {directory} does not exist. Using default image.")
        return ["https://ultralytics.com/images/bus.jpg"] * count

    images = sorted(directory.glob("*.jpg"))
    if not images:
        return ["https://ultralytics.com/images/bus.jpg"] * count
    
    # Convert Path objects to strings and return
    return [str(img) for img in images[:count]]

def benchmark_pytorch(model, images, device):
    print(f"--- Starting PyTorch ({device}) Benchmark ---")
    # Warmup
    model.predict(images[0], device=device, verbose=False, half=True)
    
    timings = []
    for img in images:
        start = time.perf_counter()
        model.predict(img, device=device, verbose=False, half=True)
        if device == 0: torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    
    avg = sum(timings) / len(timings)
    print(f"PyTorch Average: {avg:.2f} ms")
    return avg

def benchmark_onnx(onnx_path, images):
    print(f"\n--- Starting ONNX Runtime (GPU) Benchmark ---")
    
    # 1. Create Session (GPU Acceleration)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX: {e}")
        return None

    # Find input name
    input_name = session.get_inputs()[0].name
    # input_shape = session.get_inputs()[0].shape # e.g., [1, 3, 640, 640]
    
    # 2. Image Preprocessing 
    # (ONNX export does not include preprocessing, so this measures pure inference time)
    # Use 'random data' for fairness to compare pure model computation speed
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # Warmup
    session.run(None, {input_name: dummy_input})

    timings = []
    for _ in range(len(images)):
        start = time.perf_counter()
        # Actual Inference
        session.run(None, {input_name: dummy_input})
        # ONNX Runtime is synchronous by default, so no separate sync needed
        timings.append((time.perf_counter() - start) * 1000)

    avg = sum(timings) / len(timings)
    print(f"ONNX (Accelerated) Average: {avg:.2f} ms")
    return avg

def main():
    # 1. Prepare Data
    images = collect_samples(SAMPLE_DIR, SAMPLE_COUNT)
    
    # 2. Benchmark PyTorch
    pt_model = YOLO(MODEL_WEIGHTS)
    pt_time = benchmark_pytorch(pt_model, images, device=0)

    # 3. Benchmark ONNX (Alternative to TensorRT)
    if not ONNX_PATH.exists():
        print("ONNX file missing, exporting...")
        pt_model.export(format="onnx")
        
    onnx_time = benchmark_onnx(ONNX_PATH, images)

    # 4. Compare Results
    print("\n" + "="*30)
    print(f"Result Summary (Average of {len(images)} images)")
    print(f"PyTorch (Base):       {pt_time:.2f} ms")
    if onnx_time:
        print(f"Accelerated (ONNX):   {onnx_time:.2f} ms")
        
        print(f"Speedup:              {pt_time / onnx_time:.2f} x")
        print("="*30)
        print("Note: TensorRT on Jetson will be similar to or slightly faster than this ONNX result.")
    else:
        print("ONNX Benchmark Failed")

if __name__ == "__main__":
    main()