# tools/perf_stats.py
# Công cụ đo hiệu suất tạo ảnh giữa model gốc và model fine-tuned bằng LoRA
# So sánh thời gian tạo ảnh và mức tiêu thụ VRAM

import sys
import os
import time
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Hack import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core import SDXLManager

def measure_performance(manager, prompt, name):
    print(f"\n⚡ Đang đo hiệu suất: {name}...")
    
    # 1. Reset bộ nhớ để đo chính xác
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 2. Bắt đầu bấm giờ
    start_time = time.time()
    
    # 3. Chạy thử 1 ảnh
    manager.generate(
        prompt=prompt, negative_prompt="", 
        steps=30, width=1024, height=1024, 
        seed=42, num_images=1
    )
    
    # 4. Kết thúc bấm giờ
    end_time = time.time()
    duration = end_time - start_time
    
    # 5. Lấy đỉnh bộ nhớ (Peak VRAM)
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) # Đổi sang GB
    
    print(f"   ⏱️ Thời gian: {duration:.2f}s")
    print(f"   💾 VRAM Max: {max_memory:.2f}GB")
    
    return duration, max_memory

def plot_charts(base_stats, lora_stats, output_file="performance_report.png"):
    labels = ['Base Model', 'Fine-tuned (LoRA)']
    times = [base_stats[0], lora_stats[0]]
    vrams = [base_stats[1], lora_stats[1]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Biểu đồ 1: Thời gian (Thấp hơn là tốt hơn)
    ax1.bar(labels, times, color=['#3498db', '#e74c3c'])
    ax1.set_title('Tốc độ tạo ảnh (Giây) - Thấp hơn là tốt')
    ax1.set_ylabel('Giây')
    for i, v in enumerate(times):
        ax1.text(i, v + 0.5, f"{v:.2f}s", ha='center', fontweight='bold')

    # Biểu đồ 2: VRAM (Thấp hơn là tốt hơn)
    ax2.bar(labels, vrams, color=['#2ecc71', '#9b59b6'])
    ax2.set_title('Mức tiêu thụ VRAM (GB)')
    ax2.set_ylabel('GB')
    for i, v in enumerate(vrams):
        ax2.text(i, v + 0.1, f"{v:.2f}GB", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\n📊 Đã lưu biểu đồ so sánh tại: {output_file}")

def run_stats(prompt, lora_path):
    manager = SDXLManager()
    
    # 1. Đo Base Model
    manager.loader.unload_lora()
    base_stats = measure_performance(manager, prompt, "Base Model")
    
    # 2. Đo LoRA Model
    manager.loader.load_lora(lora_path)
    lora_stats = measure_performance(manager, prompt, "LoRA Model")
    
    # 3. Vẽ biểu đồ
    plot_charts(base_stats, lora_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a cat", help="Prompt để test")
    parser.add_argument("--lora_path", type=str, required=True, help="Đường dẫn file LoRA")
    args = parser.parse_args()
    
    run_stats(args.prompt, args.lora_path)