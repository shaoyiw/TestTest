# SRAFG

This is the official implementation of the paper "Enhancing Interactive Image Segmentation via Semantic Reweighting and Attention Field Guidance".

### <p align="center"> Enhancing Interactive Image Segmentation via Semantic Reweighting and Attention Field Guidance
<br>

<div align="center">
  Jianwu&nbsp;Long</a> <b>&middot;</b>
  Shaoyi&nbsp;Wang</a> <b>&middot;</b>
  Yuanqin&nbsp;Liu</a>
  <br> <br>
</div>
</br>

<div align=center><img src="assets/SRAFG.png" /></div>

### Abstract

Interactive image segmentation aims to accurately separate target objects from backgrounds through limited user interactions, enabling efficient dataset annotation and supporting deep learning applications across various domains. Traditional methods often suffer from shallow information fusion and ambiguous attention maps, leading to suboptimal performance. This paper introduces an interactive segmentation model that integrates an Adaptive Guided Fusion module to inject modulated signals into intermediate Transformer layers, ensuring sustained guidance across multiple semantic levels. Additionally, an Attention Field Supervision loss is introduced to regularize self-attention maps, promoting highly discriminative feature representations. Experiments on multiple benchmarks demonstrate superior performance, with an average reduction of 12\% in the number of clicks required to achieve 90\% IoU compared to state-of-the-art methods.

### Preparations

torch 1.8.0, torchvision 0.9.0, CUDA 11.3.

```
pip3 install -r requirements.txt
```

### Download

The datasets for training and validation can be downloaded by following: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

The pre-trained models are coming soon.

### Evaluation

Before evaluation, please download the datasets and models and configure the path in configs.yml.

The following script will start validation with the default hyperparameters:

```
python scripts/evaluate_model.py CMRefiner-V2 \
--gpu=0 \
--checkpoint=./weights/SRAFG_cocolvis.pth \
--eval-mode=cvpr \
--datasets=GrabCut,Berkeley,DAVIS,SBD
```

### Training

Before training, please download the pre-trained weights (click to download: [Segformer](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia)).

Use the following code to train a base model on coco+lvis dataset:

```
python train.py ./models/segformerB3_S2_cclvs.py \
--batch-size=6 \
--ngpus=2
```

## Acknowledgement
Here, we thank so much for these great works:  [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [FocalClick](https://github.com/XavierCHEN34/ClickSEG)
