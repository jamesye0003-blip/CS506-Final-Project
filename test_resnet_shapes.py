import torch

from train_resnet_3ch import SmallResNet, CLASS_NAMES


def test_small_resnet_output_shape():
    """
    检查 SmallResNet 在假数据上的输出形状是否正确：
    输入: (batch, 3, n_mels, T)
    输出: (batch, num_classes)
    """
    num_classes = len(CLASS_NAMES)

    # 建一个很小的模型，不需要加载任何权重
    model = SmallResNet(n_classes=num_classes, in_channels=3)
    model.eval()

    # 构造一批假数据：
    # - batch_size = 4
    # - 3 通道 (log-mel, delta1, delta2)
    # - n_mels = 64（和 CFG["n_mels"] 一致即可）
    # - T = 345（和运行日志里 ResNet 使用的 T_fixed 一致，任意正数也可以）
    dummy_input = torch.randn(4, 3, 64, 345)

    with torch.no_grad():
        logits = model(dummy_input)

    # 应该输出 (4, num_classes)
    assert logits.shape == (4, num_classes)

    # 不应该出现 NaN / Inf
    assert torch.isfinite(logits).all()


def test_small_resnet_single_sample():
    """
    再测一下 batch_size = 1 时也能正常前向传播。
    """
    num_classes = len(CLASS_NAMES)
    model = SmallResNet(n_classes=num_classes, in_channels=3)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 345)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, num_classes)
    assert torch.isfinite(logits).all()