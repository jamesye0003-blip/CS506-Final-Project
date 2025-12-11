import torch
import torch.nn as nn

# 与 YAMNet 迁移学习设置保持一致
NUM_CLASSES = 10
EMBED_DIM = 1024

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


class YAMNetMLP(nn.Module):
    """
    简单的两层 MLP，用 YAMNet 1024 维 embedding 做分类。
    这里直接在测试文件里定义，避免依赖 tensorflow/tf_hub/sklearn 等。
    """

    def __init__(self, in_dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def test_yamnet_mlp_output_shape():
    """
    检查 YAMNet 迁移学习的 MLP classifier 输出形状是否正确：
    输入: (batch, EMBED_DIM) 的 embedding
    输出: (batch, num_classes)
    """
    num_classes = len(CLASS_NAMES)

    model = YAMNetMLP(in_dim=EMBED_DIM, num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(8, EMBED_DIM)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (8, num_classes)
    assert torch.isfinite(logits).all()


def test_yamnet_mlp_single_sample():
    """
    再测一下 batch_size = 1 的情况。
    """
    num_classes = len(CLASS_NAMES)

    model = YAMNetMLP(in_dim=EMBED_DIM, num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(1, EMBED_DIM)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, num_classes)
    assert torch.isfinite(logits).all()
