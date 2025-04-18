# Thinking Temporal Automatic White Balance: Datasets, Models and Benchmarks (ACMMM 2024)
* Authors: Chunxiao Li, Shuyang Wang, Xuejing Kang, Anlong Ming*
* Affiliation: School of Computer Science (National Pilot Software Engineering School), Beijing University of Posts and Telecommunications


This paper is proposed to build a large-scale dataset and a Temporal Automatic White Balance (TAWB) method called CTANet to maintain the temporal stability of the estimated illumination.

# CTA Dataset
<p align="center">
  <img src="https://github.com/ChunxiaoLe/CTA-Dataset/blob/main/example%20images/dataset.png?raw=true" width="90%">
</p>

Dataset is available: [Mate30](https://drive.google.com/drive/folders/1Z4fS29-1YcwBi1I5C7h3TuvKUR29m-De?usp=share_link)、[Vivo](https://drive.google.com/drive/folders/1PyOOorn-aH--rAecKsiYW4hgp6ETiRRW?usp=share_link)

# Framework
<p align="center">
  <img src="https://github.com/ChunxiaoLe/CTA-Dataset/blob/main/example%20images/net.png?raw=true" width="90%">
</p>

# Results presentation
<p align="center">
  <img src="https://github.com/ChunxiaoLe/CTA-Dataset/blob/main/example%20images/visualization.png?raw=true" width="90%">
</p>


# Experiment
## Requirements
```
conda env create -f environment.yml
```

## Testing
* Pretrained models: [mate30](https://drive.google.com/drive/folders/14eNAtJTuyV7-eUHCPxq4C_gPRVLfq34u?usp=share_link)、[Vivo](https://drive.google.com/drive/folders/1k7T_ADXczIOS9qNnW-hvGn34YHqnvg8w?usp=share_link); Others are coming soon...
* Please download them and put them into the floder ./model/

### Testing images
To test images, changing '--model_pth', '--model_type' and '--data_folder' in test.py and run it.

```
python test/test.py
```

## Training
* To train the model, changing '--epochs', '-batch_size' and '--lr' in train.py and run it.
```
python train/train.py --fold_num 1 --epochs 700 --batch_size 16 --random_seed 0 --lr 0.0001
```

# Citation
please cite
```
@inproceedings{li2024thinking,
  title={Thinking Temporal Automatic White Balance: Datasets, Models and Benchmarks},
  author={Li, Chunxiao and Wang, Shuyang and Kang, Xuejing and Ming, Anlong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={7976--7984},
  year={2024}
}
```

