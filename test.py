import torch

# 1. 检查 CUDA 是否可用
is_available = torch.cuda.is_available()

print(f"CUDA (GPU) 可用性: {is_available}")

# 2. 如果可用，则可以查看更多 GPU 信息
if is_available:
    # 获取可用的 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")

    # 获取当前 GPU 设备的索引号 (默认为 0)
    current_device = torch.cuda.current_device()
    print(f"当前 GPU 设备索引: {current_device}")

    # 获取当前 GPU 设备的名称
    device_name = torch.cuda.get_device_name(current_device)
    print(f"当前 GPU 设备名称: {device_name}")

    # 尝试在 GPU 上创建一个张量
    try:
        x = torch.tensor([1.0, 2.0]).to("cuda")
        print(f"成功在 GPU 上创建张量: {x}")
        print(f"张量所在的设备: {x.device}")
    except Exception as e:
        print(f"在 GPU 上创建张量时出错: {e}")

else:
    print("PyTorch 未能检测到可用的 GPU。请检查您的安装和驱动。")