#  LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-resolution (CVPR, 2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_LAU-Net_Latitude_Adaptive_Upscaling_Network_for_Omnidirectional_Image_Super-Resolution_CVPR_2021_paper.html)

Pytorch implementation for "LAU-Net: Latitude Adaptive Upscaling Network for Omnidirectional Image Super-resolution".

**We are now fixing some bugs, please download all the resource after the reconstruction. We will finish soon.**


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


## ODI-SR dataset

Please waiting for update.

## Pretrained model

Please waiting for update.

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
My email: wang_hao@buaa.edu.cn
