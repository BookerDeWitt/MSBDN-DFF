# MSBDN-DFF
The source code of CVPR 2020 paper **"Multi-Scale Boosted Dehazing Network with Dense Feature Fusion"** by  [Hang Dong](https://sites.google.com/view/hdong/%E9%A6%96%E9%A1%B5), [Jinshan Pan](https://jspan.github.io/), [Zhe Hu](https://zjuela.github.io/), Xiang Lei, [Xinyi Zhang](http://xinyizhang.tech), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

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
We find the choices of training images play an important role during the training stage, so we offer the training set of HDF5 format: 

[Baidu Yun](https://pan.baidu.com/s/1NqAaec3MFwFU9ZM2lfR_4w) code:v8ku

You can use the DataSet_HDF5() in ./datasets/dataset_hf5.py to load these HDF5 files.


## Citation

If you use these models in your research, please cite:

	@conference{MSBDN-DFF,
		author = {Hang, Dong and Jinshan, Pan and Zhe, Hu and Xiang, Lei and Fei, Wang and Ming-Hsuan, Yang},
		title = {Multi-Scale Boosted Dehazing Network with Dense Feature Fusion},
		booktitle = {CVPR},
		year = {2020}
	}
