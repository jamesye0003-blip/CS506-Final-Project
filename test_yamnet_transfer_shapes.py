import torch
import torch.nn as nn

# Keep consistent with the YAMNet transfer learning setup
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

#A simple two-layer MLP that uses 1024-dimensional YAMNet embeddings
class YAMNetMLP(nn.Module):


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

#Verify that the output shape of the YAMNet-transfer MLP classifier is correct.
def test_yamnet_mlp_output_shape():

    num_classes = len(CLASS_NAMES)

    model = YAMNetMLP(in_dim=EMBED_DIM, num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(8, EMBED_DIM)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (8, num_classes)
    assert torch.isfinite(logits).all()

#Additional check: ensure the model works when batch_size = 1.
def test_yamnet_mlp_single_sample():

    num_classes = len(CLASS_NAMES)

    model = YAMNetMLP(in_dim=EMBED_DIM, num_classes=num_classes)
    model.eval()

    dummy_input = torch.randn(1, EMBED_DIM)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, num_classes)
    assert torch.isfinite(logits).all()
