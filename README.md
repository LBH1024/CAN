# When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition

This is the official pytorch implementation of CAN (ECCV'2022). 

>*Bohan Li, Ye Yuan, Dingkang Liang, Xiao Liu, Zhilong Ji, Jinfeng Bai, Wenyu Liu, Xiang Bai*

## Pipeline

<p align="left"><img src="https://github.com/LBH1024/CAN/blob/main/assets/img490.jpg" width="400"/></p>

## Abstract

<p align="justify">
Recently, most handwritten mathematical expression recognition (HMER) methods adopt the encoder-decoder networks, which directly predict the markup sequences from formula images with the attention mechanism. However, such methods may fail to accurately read formulas with complicated structure or generate long markup sequences, as the attention results are often inaccurate due to the large variance of writing styles or spatial layouts. To alleviate this problem, we propose an unconventional network for HMER named Counting-Aware Network (CAN), which jointly optimizes two tasks: HMER and symbol counting. Specifically, we design a weakly-supervised counting module that can predict the number of each symbol class without the symbol-level position annotations, and then plug it into a typical attention-based encoder-decoder model for HMER. Experiments on the benchmark datasets for HMER validate that both joint optimization and counting results are beneficial for correcting the prediction errors of encoder-decoder models, and CAN consistently outperforms the state-of-the-art methods. In particular, compared with an encoder-decoder model for HMER, the extra time cost caused by the proposed counting module is marginal. 
</p>

## Datasets

Download the CROHME dataset from [BaiduYun](https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg) (downloading code: 1234) and put it in ```datasets/```.

## Training

Check the config file ```config.py``` and train with the CROHME dataset:

```python train.py --dataset CROHME```.

## Testing

Fill in the ```checkpoint``` (pretrained model path) in the config file ```config.py``` and test with the CROHME dataset:

```python inference.py --dataset CROHME```.

Note that the testing dataset path is set in the ```inference.py```.
