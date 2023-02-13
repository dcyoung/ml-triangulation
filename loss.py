import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DisMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()."""

    def __init__(self, num_features, num_classes, temperature=1.0):
        super(DisMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.distance_scale, 1.0)
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.temperature = nn.Parameter(
            torch.tensor([temperature]), requires_grad=False
        )

    def forward(self, features):
        distances_from_normalized_vectors = torch.cdist(
            F.normalize(features),
            F.normalize(self.prototypes),
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        ) / math.sqrt(2.0)
        isometric_distances = (
            torch.abs(self.distance_scale) * distances_from_normalized_vectors
        )
        logits = -(isometric_distances + isometric_distances.mean(dim=1, keepdim=True))
        return logits / self.temperature

    def extra_repr(self):
        return "num_features={}, num_classes={}".format(
            self.num_features, self.num_classes
        )


class DisMaxLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""

    def __init__(self, model_classifier):
        super(DisMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0
        self.alpha = 1.0

    def forward(self, logits, targets):
        ##############################################################################
        ##############################################################################
        """Probabilities and logarithms are calculated separately and sequentially."""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss."""
        ##############################################################################
        ##############################################################################
        batch_size = logits.size(0)

        probabilities = (
            nn.Softmax(dim=1)(self.entropic_scale * logits)
            if self.model_classifier.training
            else nn.Softmax(dim=1)(logits)
        )
        probabilities_at_targets = probabilities[range(batch_size), targets]
        loss = -torch.log(probabilities_at_targets).mean()

        return loss

