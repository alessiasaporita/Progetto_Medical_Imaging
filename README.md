# Glioma Segmentation with UNet3D and nnFormer
This repository contains all materials necessary to reproduce [UNet3D](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and [nnFormer](https://github.com/282857341/nnFormer) on the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/) dataset for the glioma segmentation task.

## 📖 Introduction
Brain tumor segmentation is a key challenge in medical imaging, requiring the joint use of four imaging modalities to precisely identify tumor regions. Most existing approaches have been benchmarked mainly on the BraTS 2018 dataset, which contains only $285$ volumes. In this work, we reproduce and thoroughly analyze the most relevant models on the larger and more diverse BraTS 2023 dataset, consisting of $1,251$ volumes. Our experiments demonstrate that nnFormer achieve performance comparable to that of UNet3D on the BraTS 2023 dataset.

## 📊 Results on BraTS 2023
| Model         | ET        | TC        | WT        | Avg              |
|---------------|-----------|-----------|-----------|------------------|
| UNet3D        | **79.02** | **88.95** | **91.63** | **86.53**        |
| nnFormer      | 76.83     | 87.52     | 89.17     | 84.51            |

> ET = Enhancing Tumor, TC = Tumor Core, WT = Whole Tumor.

## 🗂️ Dataset
Before running this project, you need to download the data from BraTS 2023 Challenge, specifically the subset for [Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351) task.


## 🛠️ Installation
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
git clone https://github.com/alessiasaporita/Progetto_Medical_Imaging.git
cd Progetto_Medical_Imaging
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧩 Pipeline: UNet3D & nnFormer 
### Preprocess data
Set the data paths in `preprocess.py` and then run `python preprocess.py`.

### Training
Run the training script `train.py` with the following arguments:
```
python train.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --batch_size <BATCH_SIZE> \                  # Batch size
  --num_epochs <MAX_EPOCHS>   \                # Total number of training epochs
  --model <MODEL_NAME> \                       # UNet3D / nnFormer
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --deep_supervision                           # Enable deep supervision
  --val_check <VAL_CHECK>                      # Number of epochs between validations
```

### Test
Run the test script `test.py` with the following arguments:
```
python test.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --model <MODEL_NAME> \                       # UNet3D / nnFormer
  --savepath <OUTPUT_PATH> \                   # Directory for saving results
  --resume <RESUME_PATH> \                     # Path to the checkpoints 
  --deep_supervision                           # Set deep supervision if you have used it during training
```

## References
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
* [nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer](https://github.com/282857341/nnFormer)

