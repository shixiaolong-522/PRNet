# PRNet: Prototype Reorganization Few-Shot Semantic Segmentation Network
* Pascal-5 Benchmark with ResNet50
* Pascal-5 Benchmark with ResNet101
* COCO-20 Benchmark with ResNet50
* COCO-20 Benchmark with ResNet101

### Requirements
```
Python==3.8
GCC==5.4
torch==1.6.0
torchvision==0.7.0
tensorboardX
tqdm
PyYaml
opencv-python
pycocotools
```
#### Build Dependencies
```
cd model/ops/
bash make.sh
cd ../../
```

### Data Preparation

+ PASCAL-5^i: Please refer to [PFENet](https://github.com/dvlab-research/PFENet) to prepare the PASCAL dataset for few-shot segmentation. 

+ COCO-20^i: Please download COCO2017 dataset from [here](https://cocodataset.org/#download). Put or link the dataset to ```YOUR_PROJ_PATH/data/coco```. And make the directory like this:

## References
Our work is based on these models. (IPMT,PFENet and SSP)
* [IPMT](https://github.com/liuyuanwei98/ipmt):Intermediate Prototype Mining Transformer for Few-Shot Semantic Segmentation
* [PFENet](https://github.com/Jia-Research-Lab/PFENet):Prior Guided Feature Enrichment Network for Few-Shot Segmentation
* [SSP](https://github.com/fanq15/ssp):Self-Support Few-Shot Semantic Segmentation

