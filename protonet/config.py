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
