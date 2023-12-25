from dataclasses import dataclass


@dataclass
class EpisodeConfig:
    # number of episode iterations
    num_episodes: int
    # number of classes per episode
    nc: int
    # number of support examples per class
    ns: int
    # number of query examples per class
    nq: int


@dataclass
class DynamicEpisodeConfig(EpisodeConfig):
    # initial scaling value
    scaling: int = 1
    # scaling formula
    scaling_formula: str = "scaling+0"
    # percent formula
    percent_formula: str = "min(nc, scaling*parent)"
