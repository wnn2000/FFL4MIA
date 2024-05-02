# Fair Federated Learning for Medical Image Analysis

Welcome to the official repository for the paper titled "[From Optimization to Generalization: Fair Federated Learning against Quality Shift via Inter-Client Sharpness Matching](https://arxiv.org/abs/2404.17805)". This paper has been accepted for presentation at the IJCAI'24 main technical track.

<p align="center">
    <img src="./assets/bg.png" alt="Project Overview" width="80%"/>
</p>


## About
In this repository, we provide the implementation of our proposed **FedISM** approach, along with **other baseline methods** including [FedAvg (AISTATS'17)](https://arxiv.org/abs/1602.05629), [Agnostic-FL (ICML'19)](https://arxiv.org/abs/1902.00146), [q-FedAvg (ICLR'20)](https://arxiv.org/abs/1905.10497), [FairFed (AAAI'23)](https://arxiv.org/abs/2110.00857), [FedCE (CVPR'23)](https://arxiv.org/abs/2303.16520) and [FedGA (CVPR'23)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf).

Our goal is to advance the development of fair federated learning in medical image analysis and related fields.


## Requirements
We recommend using conda to setup the environment. See `requirements.txt` for the environment configuration.


## Datasets Preparation
Please download the ICH dataset from [kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). Please download the ISIC 2019 dataset from this [link](https://challenge.isic-archive.com/data/#2019). Data partition can be found in the paper.


## Code
Things are coming soon.


## Citation
If this repository is useful for your research, please consider citing:
```
@inproceedings{FedISM,
    title={From Optimization to Generalization: Fair Federated Learning against Quality Shift via Inter-Client Sharpness Matching},
    author={Wu, Nannan and Kuang, Zhuo and Yan, Zengqiang and Yu, Li},
    booktitle={IJCAI},
    year={2024}
}
```

## Contact
If you have any questions, please contact wnn2000@hust.edu.cn.