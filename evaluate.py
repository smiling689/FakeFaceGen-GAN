import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from network_128 import Generator # 从你的 network.py 导入生成器
from dataloader import DataGenerater # 从你的 dataloader.py 导入数据加载器

# FID 计算函数
def calculate_fid(real_features, fake_features):
    # 计算均值和协方差
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # 计算均值部分的平方差
    ssdiff = np.sum((mu_real - mu_fake)**2.0)
    
    # 计算协方差部分的迹
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    # 检查复数情况
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

# 特征提取辅助函数 (已修正)
@torch.no_grad()
def get_activations(data_loader, inception_model, device, generator_model=None):
    inception_model.eval()
    activations = []
    
    desc = ""
    # 根据是否有生成器模型来决定是处理真实图片还是生成虚假图片
    if generator_model:
        generator_model.eval()
        desc = "Generating fake images and getting activations"
    else:
        desc = "Processing real images and getting activations"

    for data in tqdm(data_loader, desc=desc):
        if generator_model:
            # 生成虚假图片
            noise_dim = 100 # G_DIMENSION
            noise = torch.randn(data_loader.batch_size, noise_dim, 1, 1, device=device)
            images = generator_model(noise)
        else:
            # 处理真实图片
            images = data.to(device)

        # InceptionV3期望输入是(N, 3, 299, 299)
        upsampled_images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 修正：总是使用 inception_model 来提取特征
        act = inception_model(upsampled_images)
        # InceptionV3 在训练时返回一个元组(logits, aux_logits)，评估时只返回 logits
        if isinstance(act, tuple):
            act = act[0]
        
        act = act.cpu().numpy().reshape(images.shape[0], -1)
        activations.append(act)
            
    return np.concatenate(activations, axis=0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载Inception-v3模型 (使用新的 'weights' API)
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    # 我们只需要特征，所以移除最后的分类层
    inception_model.fc = nn.Identity()

    # 2. 加载你的训练好的生成器
    G = Generator().to(device)
    try:
        G.load_state_dict(torch.load("generator_128.params", map_location=device))
    except FileNotFoundError:
        print("Error: generator.params not found. Please train your model first.")
        exit()
    
    # 3. 准备数据加载器
    batch_size = 32 # 减小batch size以防显存不足
    real_dataset = DataGenerater()
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    # 创建一个loader来确定生成多少张假图片
    # 它的长度与real_loader相同，但我们只用它来迭代计数
    fake_loader = DataLoader([0] * len(real_loader), batch_size=batch_size)

    # 4. 提取真实图片和生成图片的特征 (使用修正后的函数调用)
    real_activations = get_activations(real_loader, inception_model, device)
    fake_activations = get_activations(fake_loader, inception_model, device, generator_model=G)

    # 5. 计算并打印FID分数
    fid_score = calculate_fid(real_activations, fake_activations)
    print(f"\nFréchet Inception Distance (FID): {fid_score:.4f}")