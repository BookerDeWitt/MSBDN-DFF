# MSBDN-DFF
The source code of CVPR 2020 paper **"Multi-Scale Boosted Dehazing Network with Dense Feature Fusion"** by  [Hang Dong](https://sites.google.com/view/hdong/%E9%A6%96%E9%A1%B5), [Jinshan Pan](https://jspan.github.io/), [Zhe Hu](https://zjuela.github.io/), Xiang Lei, [Xinyi Zhang](http://xinyizhang.tech), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)
## Updates
(2020.12.28) Releasing the training scripts and the improved model.

## Dependencies
* Python 3.6
* PyTorch >= 1.1.0
* torchvision
* numpy
* skimage
* h5py
* MATLAB

## Test
1. Download the [Pretrained model on RESIDE](https://drive.google.com/open?id=1da13IOlJ3FQfH6Duj_u1exmZzgXPaYXe) and
[Test set](https://drive.google.com/open?id=1qZlnJN4ybjunc2BGh6kjOUfFdVxuNS-P) to  ``MSBDN-DFF/models`` and ``MSBDN-DFF/``folder, respectively.

2. Run the ``MSBDN-DFF/test.py`` with cuda on command line: 
```bash
MSBDN-DFF/$python test.py --checkpoint path_to_pretrained_model
```

3. The dehazed images will be saved in the directory of the test set.

## Train
We find the choices of training images play an important role during the training stage, so we offer the training set of HDF5 format: [Baidu Yun](https://pan.baidu.com/s/1NqAaec3MFwFU9ZM2lfR_4w) (Password:**v8ku**)

1. Download the HDF5 files to path_to_dataset.

2. Run the ``MSBDN-DFF/train.py`` with cuda on command line: 
```bash
MSBDN-DFF/$python train.py --dataset path_to_dataset/RESIDE_HDF5_all/ --lr 1e-4 --batchSize 16 --model MSBDN-DFF-v1-1 --name MSBDN-DFF
```

3.(Optional) We also provide a more advanced model (**MSBDN-RDFF**) by adopting the Relaxtion Dense Feature Fusion (**RDFF**) module. 
```bash
MSBDN-DFF/$python train.py --dataset path_to_dataset/RESIDE_HDF5_all/ --lr 1e-4 --batchSize 16 --model MSBDN-RDFF --name MSBDN-RDFF
```

By repalcing the **DFF** module with **RDFF** module, the **MSBDN-RDFF** outperforms the original **MSBDN-DFF** by a margin of 0.87 dB with less parameters on the SOTS dataset. More details will be released soon.

| Model | SOTS PSNR(dB) | Parameters |
|  :-----  |  :-----:  | :-----:  |
|  MSBDN-DFF (CVPR paper)    |  33.79  | 31M  |
|  MSBDN-RDFF (Improved)  |  34.66  | 29M  |


## Citation

If you use these models in your research, please cite:

	@conference{MSBDN-DFF,
		author = {Hang, Dong and Jinshan, Pan and Zhe, Hu and Xiang, Lei and Xinyi, Zhang and Fei, Wang and Ming-Hsuan, Yang},
		title = {Multi-Scale Boosted Dehazing Network with Dense Feature Fusion},
		booktitle = {CVPR},
		year = {2020}
	}
