from torch import Tensor
import torch.nn as nn


class SquaredL2Distance(nn.Module):
    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        # query: N x d
        # source: M x d
        dists = (
            query.unsqueeze(1) - source.unsqueeze(0)
        ).square().sum(dim=2)

        # dists: N x M
        return dists


class ProtoNet(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        dist_layer: nn.Module = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.dist_layer = dist_layer or SquaredL2Distance()

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward_train(self, query_samples: Tensor, support_samples: Tensor) -> Tensor:
        # query_samples: (Nc * Nq) x *shape
        # support_samples: Nc x Ns x *shape
        num_classes, num_support_per_class = support_samples.shape[:2]

        query_features = self.forward(query_samples)

        # query_features: (Nc * Nq) x d

        proto_features = self.forward(
            # Nc x Ns x *shape -> (Nc * Ns) x *shape
            support_samples.flatten(start_dim=0, end_dim=1)
        ).unflatten(
            dim=0, sizes=(num_classes, num_support_per_class)
        )
        # proto_features: Nc x Ns x d

        # Nc x Ns x d -> Nc x d
        proto_features = proto_features.mean(dim=1)

        dists = self.dist_layer(query_features, proto_features)
        # dists: (Nc * Nq) x Nc

        # convert distance to similarty by multiplying with -1
        return -dists
