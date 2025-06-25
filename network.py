import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        # 潜向量维度
        G_DIMENSION = 100
        # 判别器和生成器的特征图大小
        d_feature_map_size = 64

        self.main = nn.Sequential(
            # 输入: (N, 3, 64, 64)
            nn.Conv2d(3, d_feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 64, 32, 32)

            nn.Conv2d(d_feature_map_size, d_feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 128, 16, 16)

            nn.Conv2d(d_feature_map_size * 2, d_feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 256, 8, 8)

            nn.Conv2d(d_feature_map_size * 4, d_feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 512, 4, 4)

            nn.Conv2d(d_feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 输出: (N, 1, 1, 1)
            nn.Sigmoid() # 将输出压缩到 [0, 1] 之间，以配合 BCELoss
        )
    
    def forward(self, x):
        """
        前向传播
        """
        return self.main(x)
    
    

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        # 潜向量维度
        G_DIMENSION = 100
        # 判别器和生成器的特征图大小
        g_feature_map_size = 64

        self.main = nn.Sequential(
            # 输入: (N, 100, 1, 1) 维的噪声
            nn.ConvTranspose2d(G_DIMENSION, g_feature_map_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 8),
            nn.ReLU(True),
            # 输出: (N, 512, 4, 4)

            nn.ConvTranspose2d(g_feature_map_size * 8, g_feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 4),
            nn.ReLU(True),
            # 输出: (N, 256, 8, 8)

            nn.ConvTranspose2d(g_feature_map_size * 4, g_feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 2),
            nn.ReLU(True),
            # 输出: (N, 128, 16, 16)

            nn.ConvTranspose2d(g_feature_map_size * 2, g_feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size),
            nn.ReLU(True),
            # 输出: (N, 64, 32, 32)
            
            nn.ConvTranspose2d(g_feature_map_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # 输出: (N, 3, 64, 64)
            nn.Tanh() # 将输出压缩到 [-1, 1] 之间
        )

    
    def forward(self, x):
        """
        前向传播
        """
        return self.main(x)