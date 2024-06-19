import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SquaredL2Distance(nn.Module):
    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        # query: N x d
        # source: M x d
        dists = (query.unsqueeze(1) - source.unsqueeze(0)).square().sum(dim=2)

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

    def forward_train(
        self,
        query_samples: Tensor,
        support_samples: Tensor,
    ) -> Tensor:
        # query_samples: (Nc * Nq) x *shape
        # support_samples: Nc x Ns x *shape
        num_classes, num_support_per_class = support_samples.shape[:2]

        query_features = self.forward(query_samples)

        # query_features: (Nc * Nq) x d

        proto_features = self.forward(
            # Nc x Ns x *shape -> (Nc * Ns) x *shape
            support_samples.flatten(start_dim=0, end_dim=1)
        ).unflatten(dim=0, sizes=(num_classes, num_support_per_class))
        # proto_features: Nc x Ns x d

        # Nc x Ns x d -> Nc x d
        proto_features = proto_features.mean(dim=1)

        dists = self.dist_layer(query_features, proto_features)
        # dists: (Nc * Nq) x Nc

        # convert distance to similarty by multiplying with -1
        return -dists


class TapNet(ProtoNet):
    def __init__(
        self,
        encoder: nn.Module,
        dist_layer: nn.Module = None,
        embed_dim: int = None,
        project_dim: int = None,
        num_classes: int = None,
    ):
        project_dim = project_dim or embed_dim - num_classes
        assert (
            project_dim + num_classes <= embed_dim
        ), "projection dimension + number of classes must be less than feature dimensions"
        super().__init__(encoder, dist_layer=dist_layer)

        self.cls_ref_phi = nn.Linear(embed_dim, num_classes)
        self.register_buffer("normalizer", torch.tensor(num_classes - 1))
        self.project_dim = project_dim
        self.num_classes = num_classes
        self.M = None

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x) @ self.M

    def forward_train(
        self,
        query_samples: Tensor,
        support_samples: Tensor,
    ) -> Tensor:
        """_summary_

        Args:
            query_samples (Tensor): (Nc * Nq) x *shape
            support_samples (Tensor): Nc x Ns x *shape

        Returns:
            Tensor: (Nc * Nq) x Nc
        """
        # query_samples:
        # support_samples:
        num_classes, num_support_per_class = support_samples.shape[:2]

        query_features = self.encoder(query_samples)

        # query_features: (Nc * Nq) x d_embed

        proto_features = self.encoder(
            # Nc x Ns x *shape -> (Nc * Ns) x *shape
            support_samples.flatten(start_dim=0, end_dim=1)
        ).unflatten(dim=0, sizes=(num_classes, num_support_per_class))
        # proto_features: Nc x Ns x d_embed

        # Nc x Ns x d_embed -> Nc x d_embed
        proto_features = proto_features.mean(dim=1)

        M = self.null_space(proto_features)
        # M: d_embed x d_proj
        self.M = M

        query_features = query_features @ M
        # query_features: Nc x Ns x d_proj
        ref_proto_features = self.cls_ref_phi.weight @ M
        # ref_proto_features: Nc x d_proj

        dists = self.dist_layer(query_features, ref_proto_features)
        # dists: (Nc * Nq) x Nc

        # convert distance to similarty by multiplying with -1
        return -dists

    @torch.no_grad()
    def null_space(self, proto_features: Tensor) -> Tensor:
        """_summary_

        Args:
            proto_features (Tensor): Nc x d_embed

        Returns:
            Tensor: d_embed x d_proj
        """
        cls_ref_phi = self.cls_ref_phi.weight

        cls_ref_phi_hat = (self.num_classes * cls_ref_phi) - cls_ref_phi.sum(
            dim=0, keepdims=True
        )
        cls_ref_phi_hat = cls_ref_phi_hat / self.normalizer
        # cls_ref_phi_hat: Nc x d_embed

        epsilon = F.normalize(cls_ref_phi_hat, dim=1) - F.normalize(
            proto_features, dim=1
        )
        # epsilon: Nc x d_embed

        _, _, Vh = torch.linalg.svd(epsilon)
        # Vh: d_embed x d_embed

        return Vh[self.num_classes : self.num_classes + self.project_dim, :].T
