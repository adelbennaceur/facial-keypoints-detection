import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet


def efficientnet_modified():
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Sequential(
        nn.Linear(in_features=1280, out_features=1024, bias=True),
        nn.Linear(in_features=1280, out_features=68, bias=True),
    )

    return model
