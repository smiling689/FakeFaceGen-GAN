import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from network import Generator, Discriminator # 确保 Discriminator 是移除了 Sigmoid 的版本
from dataloader import DataGenerater

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # 生成随机的插值权重
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
    # 在真实样本和虚假样本之间进行插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 得到判别器对插值样本的打分
    d_interpolates = critic(interpolates)
    
    # 创建一个与判别器输出形状完全相同的梯度张量
    # 这是为了匹配 autograd.grad 的输入要求
    grad_outputs = torch.ones_like(d_interpolates, device=device, requires_grad=False)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs, # 使用形状匹配的梯度
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if __name__ == '__main__':
    # 超参数
    lr = 0.0001
    beta1 = 0.0
    beta2 = 0.9
    epochs = 10 # WGAN-GP可能需要更多epochs
    batch_size = 64
    g_dim = 100
    lambda_gp = 10 # 梯度惩罚的系数
    n_critic = 5 # 每训练一次生成器，训练5次判别器

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型
    netG = Generator().to(device)
    netD = Discriminator().to(device) # Critic

    # 优化器
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    # 数据加载
    dataset = DataGenerater()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    print("Starting WGAN-GP Training...")
    for epoch in range(epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):

            # ---------------------
            #  训练判别器 (Critic)
            # ---------------------

            netD.zero_grad()

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            noise = torch.randn(b_size, g_dim, 1, 1, device=device)
            fake = netG(noise)

            # 在真实和虚假图像上得到分数
            real_output = netD(real_cpu).view(-1)
            fake_output = netD(fake.detach()).view(-1)

            # 计算梯度惩罚
            gradient_penalty = compute_gradient_penalty(netD, real_cpu.data, fake.data, device)

            # Critic 损失 = 假图分数 - 真图分数 + 梯度惩罚
            errD = torch.mean(fake_output) - torch.mean(real_output) + lambda_gp * gradient_penalty
            errD.backward()
            optimizerD.step()

            # -----------------
            #  训练生成器
            # -----------------
            # 每 n_critic 次 D 的迭代，才训练一次 G
            if i % n_critic == 0:
                netG.zero_grad()

                # 重新生成假图并通过 Critic
                noise = torch.randn(b_size, g_dim, 1, 1, device=device)
                fake = netG(noise)
                output = netD(fake).view(-1)

                # 生成器损失 = -假图分数
                errG = -torch.mean(output)
                errG.backward()
                optimizerG.step()

                # 打印损失
                if i % 100 == 0:
                    tqdm.write(f'[{epoch}/{epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    # 保存模型
    torch.save(netG.state_dict(), "generator_wgan_gp.params") 
    print("WGAN-GP training finished. Model saved as generator_wgan_gp.params")