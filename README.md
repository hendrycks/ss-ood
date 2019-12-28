# Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty

This repository contains the dataset and some code for the paper [Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1906.12340) by Dan Hendrycks, Mantas Mazeika, Saurav Kadavath, and Dawn Song.

We show that self-supervised learning can tremendously improve out-of-distribution detection as well as various types of robustness.

<img align="center" src="not_hotdog.png" width="750">

Download the one class ImageNet test set [here](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view?usp=sharing). The one class ImageNet training set is [here](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view?usp=sharing).

The code requires PyTorch 1.0 + and Python 3+.

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2019selfsupervised,
      title={Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty},
      author={Dan Hendrycks and Mantas Mazeika and Saurav Kadavath and Dawn Song},
      journal={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2019}
    }
