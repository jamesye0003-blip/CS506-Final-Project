import torch
import torch.nn as nn

# Class list consistent with the main project
CLASS_NAMES = [
    "air_conditioner",   # 0
    "car_horn",          # 1
    "children_playing",  # 2
    "dog_bark",          # 3
    "drilling",          # 4
    "engine_idling",     # 5
    "gun_shot",          # 6
    "jackhammer",        # 7
    "siren",             # 8
    "street_music",      # 9
]


# ===== Simplified copies of BasicBlock and SmallResNet =====
# Depends only on torch

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = BasicBlock(32, 64, stride=2)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (B, 3, 64, T)
        x = self.stem(x)     # (B, 32, 64, T)
        x = self.layer1(x)   # (B, 64, 32, T/2)
        x = self.layer2(x)   # (B, 128, 16, T/4)
        x = self.layer3(x)   # (B, 256, 8, T/8)

        x = self.gap(x)      # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.dropout(x)
        x = self.fc(x)       # (B, n_classes)
        return x


def test_small_resnet_output_shape():
    """
        Verify that SmallResNet produces the correct output shape on dummy data.
        Input:  (batch, 3, n_mels, T)
        Output: (batch, num_classes)
        """
    num_classes = len(CLASS_NAMES)

    model = SmallResNet(n_classes=num_classes, in_channels=3)
    model.eval()

    # Same setting as the main project: n_mels=64, T=345
    dummy_input = torch.randn(4, 3, 64, 345)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4, num_classes)
    assert torch.isfinite(logits).all()


def test_small_resnet_single_sample():
    """
    Additional test: ensure forward pass works when batch_size = 1.
    """
    num_classes = len(CLASS_NAMES)

    model = SmallResNet(n_classes=num_classes, in_channels=3)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 345)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, num_classes)
    assert torch.isfinite(logits).all()
