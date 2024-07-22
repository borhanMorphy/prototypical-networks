# ProtoNet

**&mdash; PyTorch implementations of the ProtoNet variations**

<!-- TODO add badges -->

<img src="/resources/prototypes.png"/>

## Contents

- [Installation](#installation)
- [Supported ProtoNets](#supported-protonets)
- [References](#references)
- [Citations](#citations)

## Installation

```
git clone https://github.com/borhanMorphy/prototypical-networks.git
cd prototypical-networks
pip install .
```

## Supported ProtoNets

|       Name       | Extra Parameters |  Training Metric  | Inference Metric  |
| :--------------: | :--------------: | :---------------: | :---------------: |
|   **ProtoNet**   |        No        | squared euclidean | squared euclidean |
| **SEN-ProtoNet** |        No        |        sen        |      cosine       |
|    **TapNet**    |       Yes        | squared euclidean | squared euclidean |

## References

### ProtoNet

- [Official Implementation](https://github.com/jakesnell/prototypical-networks)
- [Paper](https://arxiv.org/pdf/1703.05175.pdf)

### SEN ProtoNet

- [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680120.pdf)

### TapNet

- [Official Implementation](https://github.com/istarjun/TapNet)
- [Paper](https://arxiv.org/pdf/1905.06549.pdf)

## Citations

```bibtex
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
```
