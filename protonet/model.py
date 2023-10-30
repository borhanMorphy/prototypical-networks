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

    def forward_train(self, query_samples: Tensor, support_samples: Tensor):
        # query_samples: (Nq * Nc) x *shape
        # support_samples: Ns x Nc x *shape
        num_support_per_class, num_classes = support_samples.shape[:2]

        query_features = self.forward(query_samples)

        # query_features: (Nq * Nc) x d

        proto_features = self.forward(
            # Ns x Nc x *shape -> (Ns * Nc) x *shape
            support_samples.flatten(start_dim=0, end_dim=1)
        ).unflatten(
            dim=0, sizes=(num_support_per_class, num_classes)
        )
        # proto_features: Ns x Nc x d

        # Ns x Nc x d -> Nc x d
        proto_features = proto_features.mean(dim=0)

        dists = self.dist_layer(query_features, proto_features)
        # dists: (Nc * Nq) x Nc

        # convert distance to similarty by multiplying with -1
        return -dists
