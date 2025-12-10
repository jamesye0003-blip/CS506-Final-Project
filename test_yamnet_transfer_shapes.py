import torch

# 按你自己的文件和类名改这里：
# 例子假设 train_yamnet_transfer.py 中有:
#   - YamNetMLP (或 YAMNetMLP)
#   - CLASS_NAMES (长度为 10)
#   - EMBED_DIM = 1024
from train_yamnet_transfer import YAMNetMLP, CLASS_NAMES

EMBED_DIM = 1024
def test_yamnet_mlp_output_shape():
    """
    检查 YAMNet 迁移学习的 MLP classifier 输出形状是否正确：
    输入: (batch, EMBED_DIM) 的 embedding
    输出: (batch, num_classes)
    """
    num_classes = len(CLASS_NAMES)

    model = YAMNetMLP(in_dim=EMBED_DIM, num_classes=num_classes)
    model.eval()

    # 构造一批假 embedding：batch_size = 8, dim = EMBED_DIM
    dummy_emb = torch.randn(8, EMBED_DIM)

    with torch.no_grad():
        logits = model(dummy_emb)

    assert logits.shape == (8, num_classes)
    assert torch.isfinite(logits).all()


def test_yamnet_mlp_single_sample():
    """
    再测一下 batch_size = 1 的情况。
    """
    num_classes = len(CLASS_NAMES)
    model = YAMNetMLP(in_dim=EMBED_DIM, num_classes=num_classes)
    model.eval()

    dummy_emb = torch.randn(1, EMBED_DIM)

    with torch.no_grad():
        logits = model(dummy_emb)

    assert logits.shape == (1, num_classes)
    assert torch.isfinite(logits).all()