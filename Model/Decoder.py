import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    
    def __init__(self, in_channels=768, final_channels=3):  # (B, 768, 8, 8)
        super(Decoder, self).__init__()
        
        self.layers = nn.Sequential(
            # Upsample Layer 1: 32x32 -> 64x64, 768 -> 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            # Additional conv layers at this size
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            # Upsample Layer 2: 64x64 -> 128x128, 256 -> 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            # Additional conv layer at this size
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            
            # Upsample Layer 3: 128x128 -> 256x256, 128 -> 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            # Additional conv layer at this size
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            
            # Upsample Layer 4: 256x256 -> 256x256, 64 -> 3
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, final_channels, kernel_size=3, stride=1, padding=0),
        )
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, 32, 32)

        output = self.layers(x)

        output = torch.sigmoid(output)
 
        return output

if __name__ == "__main__":
    Y = Decoder()
    Y = Y.forward(torch.randn(2, 1024, 768))

    print(Y.shape)