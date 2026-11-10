# training/model_pilotnet.py

import torch
import torch.nn as nn


class PilotNet(nn.Module):
    """
    자율주행 RC카용 소형 CNN (PilotNet 기반)
    - 입력 : (B, 3, H, W)
    - 출력 : (B, num_classes)  (각도 분류용)
    """
    def __init__(self, num_classes: int = 5, input_shape=(3, 66, 200)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # input_shape을 기반으로 Flatten 후 차원 자동 계산
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.features(dummy)
            self.flatten_dim = feat.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
