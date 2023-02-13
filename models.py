import torch
from torch import nn
from torch import Tensor
from loss import DisMaxLossFirstPart
from scipy.special import softmax as softmax_np
import numpy as np
import numpy.typing as npt


def mmles_torch(logits: Tensor) -> Tensor:
    """Maximum Mean Logit Entropy Score"""
    probabilities = nn.Softmax(dim=1)(logits)
    return (
        logits.max(dim=1)[0]
        + logits.mean(dim=1)
        + (probabilities * torch.log(probabilities)).sum(dim=1)
    )


def mmles_np(logits: npt.NDArray) -> npt.NDArray:
    """Maximum Mean Logit Entropy Score"""
    probabilities = softmax_np(logits, axis=1)
    return (
        logits.max(1) + logits.mean(1) + (probabilities * np.log(probabilities)).sum(1)
    )


class MlpTriangulationModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_size: int = 64,
        n_hidden_layers: int = 5,
        b_norm: bool = True,
    ):
        super().__init__()
        fan_in_out = [(input_size, hidden_layer_size)] + (n_hidden_layers - 1) * [
            (hidden_layer_size, hidden_layer_size)
        ]
        layers = [
            # Flatten the data for each node into a single vector like so: [x1,y1,ss1, x2,y2,ss2...]
            nn.Flatten(start_dim=1)
        ]
        for fan_in, fan_out in fan_in_out:
            layers += (
                [nn.Linear(fan_in, fan_out, bias=not b_norm)]
                + ([nn.BatchNorm1d(fan_out)] if b_norm else [])
                + [nn.ReLU()]
            )
        self.classifier = DisMaxLossFirstPart(
            num_features=fan_in_out[-1][-1], num_classes=output_size
        )
        layers.append(self.classifier)
        self.layers = nn.Sequential(*layers)

    def forward(self, samples: Tensor) -> Tensor:
        return self.layers(samples)

    def get_softmax_scores_for_logits(self, logits: Tensor) -> Tensor:
        return nn.Softmax(dim=1)(logits)

    def predict_softmax_scores(self, samples: Tensor) -> Tensor:
        logits = self.forward(samples)
        return self.get_softmax_scores_for_logits(logits)
