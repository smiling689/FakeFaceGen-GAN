import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        d_feature_map_size = 64

        self.main = nn.Sequential(
            # 输入: (N, 3, 128, 128)
            nn.Conv2d(3, d_feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 64, 64, 64)

            nn.Conv2d(d_feature_map_size, d_feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 128, 32, 32)

            nn.Conv2d(d_feature_map_size * 2, d_feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 256, 16, 16)

            nn.Conv2d(d_feature_map_size * 4, d_feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 512, 8, 8)
            
            # --- 新增层 ---
            nn.Conv2d(d_feature_map_size * 8, d_feature_map_size * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feature_map_size * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (N, 1024, 4, 4)

            nn.Conv2d(d_feature_map_size * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 输出: (N, 1, 1, 1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)
    
    

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        G_DIMENSION = 100
        g_feature_map_size = 64

        self.main = nn.Sequential(
            # 输入: (N, 100, 1, 1) 
            nn.ConvTranspose2d(G_DIMENSION, g_feature_map_size * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 16),
            nn.ReLU(True),
            # 输出: (N, 1024, 4, 4)

            nn.ConvTranspose2d(g_feature_map_size * 16, g_feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 8),
            nn.ReLU(True),
            # 输出: (N, 512, 8, 8)

            nn.ConvTranspose2d(g_feature_map_size * 8, g_feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 4),
            nn.ReLU(True),
            # 输出: (N, 256, 16, 16)

            nn.ConvTranspose2d(g_feature_map_size * 4, g_feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size * 2),
            nn.ReLU(True),
            # 输出: (N, 128, 32, 32)
            
            nn.ConvTranspose2d(g_feature_map_size * 2, g_feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_feature_map_size),
            nn.ReLU(True),
            # 输出: (N, 64, 64, 64)
            
            nn.ConvTranspose2d(g_feature_map_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # 输出: (N, 3, 128, 128)
            nn.Tanh()
        )

    
    def forward(self, x):
        return self.main(x)