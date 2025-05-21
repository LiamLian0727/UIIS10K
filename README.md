# UIIS10K
![issues](https://img.shields.io/github/issues/LiamLian0727/UIIS10K)
![forks](https://img.shields.io/github/forks/LiamLian0727/UIIS10K?style=flat&color=orange)
![stars](https://img.shields.io/github/stars/LiamLian0727/UIIS10K?style=flat&color=red)

This repository is the official implementation of "[UWSAM: Segment Anything Model Guided Underwater Instance Segmentation and A Large-scale Benchmark Dataset]()".

If you found this project useful, please give us a star ⭐️ or [cite](#citation) us in your paper, this is the greatest support and encouragement for us.

## :speech_balloon: Updates
🚩 **News** (2025.05) We propose a large-scale underwater instance segmentation dataset, [UIIS10K](#datasets), which includes 10,048 images with pixel-level annotations for 10 categories.

## Datasets
The dataset follows the COCO format and is organized as follows:
```
    data
      ├── UIIS10K
      |   ├── annotations
      │   │   ├── multiclass_train.json
      │   │   ├── multiclass_test.json
      │   ├── background_img
      │   │   ├── background_0001.jpg
      │   │   ├── ...
      │   ├── img
      │   │   ├── train_00001.jpg
      │   │   ├── ...
      │   │   ├── test_00001.jpg
      │   │   ├── ...
```
you can get our UIIS10K dataset in [Baidu Disk](https://pan.baidu.com/s/1WwDu_jYV8JsPvOGA2l6raQ?pwd=UIIS) (pwd:UIIS) or [Google Drive](https://drive.google.com/file/d/1MYQwWrQW_n9N-q_VPMuQaroIp5gS2f-u/view?usp=sharing).

## Datasets
Code is coming soon

### Citation
If you find our repo useful for your research, please cite us:
```
@InProceedings{UIIS_Dataset_2023,
    author    = {Shijie Lian, Hua Li, Runmin Cong, Suqi Li, Wei Zhang, Sam Kwong},
    title     = {WaterMask: Instance Segmentation for Underwater Imagery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {1305-1315}
}

@article{UIIS10K_Dataset_2025,
    author    = {Hua Li, Shijie Lian, Zhiyuan Li, Runmin Cong, Sam Kwong},
    title     = {UWSAM: Segment Anything Model Guided Underwater Instance Segmentation and A Large-scale Benchmark Dataset},
    year      = {2025},
    journal   = {arXiv preprint},
}
```
## Acknowledgement
This repository is implemented based on the [MMDetection](https://github.com/open-mmlab/mmdetection) framework and [Segment Anything Model](https://huggingface.co/facebook/sam-vit-huge). Thanks to them for their excellent work.

## ⭐ Stargazers
[![Stargazers repo roster for @LiamLian0727/USIS10K](https://reporoster.com/stars/LiamLian0727/UIIS10K)](https://github.com/LiamLian0727/USIS10K/stargazers)
