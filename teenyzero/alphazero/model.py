import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaNet(nn.Module):
    def __init__(self, num_res_blocks=10, channels=128):
        super().__init__()
        
        # FIX 1: Set in_channels to 13 (6 white pieces, 6 black, 1 turn)
        self.conv_in = nn.Conv2d(13, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.pol_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(32)
        
        # FIX 2: Set output to 4672 (AlphaZero move planes: 64 squares * 73 directions)
        self.pol_fc = nn.Linear(32 * 8 * 8, 4672) 
        
        # Value Head
        self.val_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.val_fc2 = nn.Linear(128, 1)

        # Zero-init the final value layer to start at a 50/50 win probability
        nn.init.zeros_(self.val_fc2.weight)
        nn.init.zeros_(self.val_fc2.bias)

    def forward(self, x):
        # x shape: (batch, 13, 8, 8)
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy Head
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.pol_fc(p) # Raw logits (Softmax is handled in the Evaluator)
        
        # Value Head
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v)) # Output range: [-1, 1]
        
        return p, v