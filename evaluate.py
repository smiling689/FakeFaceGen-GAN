import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from network_128 import Generator
from dataloader import DataGenerater

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

# 特征提取辅助函数
@torch.no_grad()
def get_activations(data_loader, inception_model, device, generator_model=None):
    inception_model.eval()
    activations = []
    
    desc = ""
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

        upsampled_images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        act = inception_model(upsampled_images)
        if isinstance(act, tuple):
            act = act[0]
        
        act = act.cpu().numpy().reshape(images.shape[0], -1)
        activations.append(act)
            
    return np.concatenate(activations, axis=0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载Inception-v3模型
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    # 移除分类层
    inception_model.fc = nn.Identity()

    # 加载生成器
    G = Generator().to(device)
    try:
        G.load_state_dict(torch.load("generator_128.params", map_location=device))
    except FileNotFoundError:
        print("Error: generator.params not found. Please train your model first.")
        exit()
    
    # 数据加载器
    batch_size = 32 
    real_dataset = DataGenerater()
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    # 创建一个loader来确定生成多少张假图片
    fake_loader = DataLoader([0] * len(real_loader), batch_size=batch_size)

    real_activations = get_activations(real_loader, inception_model, device)
    fake_activations = get_activations(fake_loader, inception_model, device, generator_model=G)

    # 计算FID
    fid_score = calculate_fid(real_activations, fake_activations)
    print(f"\nFréchet Inception Distance (FID): {fid_score:.4f}")