import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, scale_factor=2, num_frames=3, channels=3):
        super(ESPCN, self).__init__()
        
        input_channels = channels * num_frames
        
        self.act = nn.PReLU()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(32, channels * (scale_factor ** 2), kernel_size=3, padding=1)
        
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)