# Multi-attention

This is the official code repository for "MAIN: A Multi-Attention Integration Network for Robust Semantic Segmentation Across Diverse Remote Sensing Datasets". 
  
## Abstract  
Employing deep learning for land cover feature recognition in remote sensing (RS) images is a widely adopted and effective approach. However, most existing methods face challenges in maintaining consistent performance across datasets with diverse characteristics. Consequently, the development of models that are generalizable across multiple datasets has become an essential area of research. Inspired by the growing prominence of attention mechanisms, we propose the Multi-Attention Integration Network (MAIN), which incorporates three distinct attention mechanisms into the encoder and effectively integrates merged features with the decoder. Additionally, we introduce a new remote sensing dataset, CF-FD, and perform experiments on five datasets with varying characteristics for model comparison. The results demonstrate that our model exhibits strong performance and robust adaptability across different datasets. Notably, it achieves the highest Overall Accuracy (OA) among all compared models on four out of the five datasets. The MAIN model shows significant ability for generalization and adaptation to diverse remote sensing datasets.

## Environment
Create an new conda virtual environment
```bash
conda create -n xxxx python=3.9.19 -y
conda activate xxxx
```
Clone this repo and install required packages:
```bash
git clone https://github.com/yaw6622/Multi-attention
cd Multi-attention/
pip install -r requirements.txt
```
## Comparsion with some other models
Due to the lack of generality in some code frameworks, we have not fully transferred all code implementations to our provided framework. At this stage, certain comparative experiments are not conducted within our framework. The models implemented within our framework include the MAIN model, FPN model, DeeplabV3+ and LinkNet model.
### Segmenter
For the Segmenter model, experiments are conducted using the codebase available at [Segmenter](https://github.com/rstrudel/segmenter).

### Vim
For the Vim model, experiments are conducted using the codebase available at [Vim](https://github.com/hustvl/Vim).

### Segformer
For the Segformer model, experiments are conducted using the codebase available at [Segformer](https://github.com/NVlabs/SegFormer).

## Train the model
```bash
cd Multi-attention
python My_.py  # Train the Model on the datasets.
```trainFrame

## Dataset availability
See Dataset.txt.

### CF-FD

### Open EarthMap

### Potsdam

### Open EarthMap

### Open EarthMap

## Obtain the outputs  
- After trianing, you could obtain the results in './results/'



## Pretrained model

## Acknowledgments  
  
- We thank the authors of [Segformer](https://github.com/NVlabs/SegFormer),  [Vmamba](https://github.com/MzeroMiko/VMamba) and [EMA](https://github.com/YOLOonMe/EMA-attention-module) for their open-source codes.
