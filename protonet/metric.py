import torch
import torch.nn as nn
from torch import Tensor


class SquaredL2Distance(nn.Module):
    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        """Squared euclidean distance

        Args:
            query (Tensor): stacked query vectors with shape of N x d
            source (Tensor): stacked source vectors with shape of M x d

        Returns:
            Tensor: distance measure with shape of N x M
        """
        dists = (query.unsqueeze(1) - source.unsqueeze(0)).square().sum(dim=2)

        return dists


class SEN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_query: int,
        neg_epsilon: float,
        pos_epsilon: float,
    ) -> None:
        super().__init__()
        self.dist_se = SquaredL2Distance()
        epsilon = torch.eye(num_classes)
        # fill everywhere with negative epsilon
        epsilon.fill_(neg_epsilon)
        # fill diagonal with positive epsilon
        epsilon.fill_diagonal_(pos_epsilon)
        # repeat positive epsilon for each query
        epsilon = epsilon.repeat_interleave(num_query, dim=0)
        # epsilon: (Nc * Nq) x Nc

        self.register_buffer("epsilon", epsilon)

    def dist_norm(self, query: Tensor, source: Tensor) -> Tensor:
        """distance of noramlized

        Args:
            query (Tensor): stacked query vectors with shape of N x d
            source (Tensor): stacked source vectors with shape of M x d

        Returns:
            Tensor: N x M
        """
        query_norm = query.norm(dim=1)
        # query_norm: N,
        source_norm = source.norm(dim=1)
        # source_norm: M,
        dists = (query_norm.unsqueeze(1) - source_norm.unsqueeze(0)).square()

        return dists

    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        """SEN Dissimilarity measurement

        Args:
            query (Tensor): stacked query vectors with shape of N x d (where N is assumed to be Nc * Nq)
            source (Tensor): stacked source vectors with shape of M x d (where M is assumed to be Nc)

        Returns:
            Tensor: N x M
        """
        se_dist = self.dist_se(query, source)
        # se_dist: N x M
        n_dist = self.dist_norm(query, source)
        # n_dist: N x M

        return (se_dist + self.epsilon * n_dist).sqrt()
