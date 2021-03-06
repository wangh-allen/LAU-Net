#  LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-resolution (CVPR, 2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_LAU-Net_Latitude_Adaptive_Upscaling_Network_for_Omnidirectional_Image_Super-Resolution_CVPR_2021_paper.html)

Pytorch implementation for "LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-resolution".


## Dependencies
```
Python=3.7, PyTorch=1.7.0, numpy, skimage, cv2, matplotlib, tqdm
```

## Test scripts
Put dataset in ./dataset.

Put pre-trained model in ./results/model.

Run the main.py:
```bash
python main.py --test_only --datastest=test --pre_train='./results/model/model_best.pt'
```


## ODI-SR and testing datasets

Google Drive:

link: https://drive.google.com/drive/folders/1w7m1r-yCbbZ7_xMGzb6IBplPe4c89rH9?usp=sharing

## Pretrained model

Google Drive:

link: https://drive.google.com/drive/folders/15FxJOB0hWR3WZTg9CNxKKjGciQF8ZwSK?usp=sharing

## Citation

If you find our paper or code useful for your research, please cite:

```
@InProceedings{Deng_2021_CVPR,
    author    = {Deng, Xin and Wang, Hao and Xu, Mai and Guo, Yichen and Song, Yuhang and Yang, Li},
    title     = {LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9189-9198}
}
```

## Contact

If you have any problem, please contact with me through email. I will reply soon.

My email: wang_hao@buaa.edu.cn
