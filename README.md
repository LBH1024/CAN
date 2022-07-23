# When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition

The official pytorch implementation of CAN (ECCV'2022).

## Datasets

Download the CROHME dataset from [BaiduYun](https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg) (downloading code: 1234) and put it in ```datasets/```

## Training

Check the config file ```config.py``` and train with the CROHME dataset:

```python train.py --dataset CROHME```

## Testing

Fill the ```checkpoint``` (pretrained model path) in the config file ```config.py``` and test with the CROHME dataset:

```python inference.py --dataset CROHME```

Note that the testing dataset path is set in the ```inference.py```
