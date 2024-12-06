import torch
import torch.nn as nn

class C3D(nn.Module):
    """
    The C3D network for feature extraction.
    """
    # TODO 有问题

    def __init__(self, in_channels, out_channels):
        super(C3D, self).__init__()

         # 减少卷积层数，保持简单
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # self.conv2 = nn.Conv3d(out_channels, out_channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.__init_weight()

    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, length, period = x.shape
        # x的形状为[B, N, length of period, period]
        # 需要调整为[C, D, H, W]，即[B, N, length, width]
        # 在这里我们假设length of period对应于depth，period对应于height和width
        x = x.permute(0, 1, 2, 3).unsqueeze(2)  # [B, N, 1, length, period] 这里添加一个维度作为depth
        x = x.repeat(1, 1, N, 1, 1) 
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        # x = self.relu(self.conv2(x))
        # x = self.pool2(x)
        

        # x的形状为[B, 512, D, H, W]，需要将其调整回[B, N, length of period, period]
        x = x.squeeze(2)  # 去掉depth维度
        return x.permute(0, 2, 1, 3)  # 确保输出形状为[B, N, length， period]
    
    