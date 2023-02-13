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
        # layers.append(nn.Linear(fan_in_out[-1][-1], output_size))
        # parameter init
        # with torch.no_grad():
        #     layers[-1].weight *= 0.1  # make last layer less confident
        layers.append(
            DisMaxLossFirstPart(
                num_features=fan_in_out[-1][-1], num_classes=output_size
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, samples: Tensor) -> Tensor:
        return self.layers(samples)
