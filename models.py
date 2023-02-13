from torch import nn
from torch import Tensor
from loss import DisMaxLossFirstPart


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

    def get_mmles_scores_for_logits(self, logits: Tensor) -> Tensor:
        return self.classifier.mmles_scores(logits)

    def predict_mmles_scores(self, samples: Tensor) -> Tensor:
        """Maximum Mean Logit Entropy Score"""
        logits = self.forward(samples)
        return self.get_mmles_scores_for_logits(logits)
