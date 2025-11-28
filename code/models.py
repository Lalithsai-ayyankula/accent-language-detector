# models.py
import torch.nn as nn

class SimpleMFCCCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, n_mfcc=40, max_frames=200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2))
        )
        # compute flatten dims manually or with example input; assume pooling reduces by 4
        h = max(1, n_mfcc // 4)
        w = max(1, max_frames // 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * h * w, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
