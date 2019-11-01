## LassoNet: Deep Lasso-Selection of 3D Point Clouds

### Citation
If you find our work useful in your research, please consider citing:

```
@article{chen2019b,
  title={LassoNet: Deep Lasso-Selection of 3D Point Clouds},
  author={Chen, Zhutian and Zeng, Wei and Yang, Zhiguang and Yu, Lingyun and Fu, Chi-Wing and Qu, Huamin},
  journal = {{IEEE Transactions on Visualization and Computer Graphics}},
  year    = {2019}, 
  volume  = {}, 
  number  = {}, 
  pages   = {1-1}
}
```

### Introduction
This project, based on our IEEE VIS'19 paper, aims at learning a latent mapping from viewpoint and lasso to point cloud regions using a deep learning model. This project is builted based on [pointnet](https://github.com/charlesq34/pointnet) / [pointnet++](https://github.com/charlesq34/pointnet2). You can also check our [project webpage](https://lassonet.github.io/) for more details.

In this repository, we release code and data for training the deep learning model in our method.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.12 GPU version and Python 3.6 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing like `h5py`, etc.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.12.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
```
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Dataset
The selection datasets on ShapeNet: [d1](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zchenbn_connect_ust_hk/EbA6lKsGIJ9OnXpcosV3TnEBzH84-AKzGimMGPEmVuIiRA?e=fFatzD).
The selection datasets on S3DIS: [d2](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zchenbn_connect_ust_hk/EU34qjiMZRdPmPgy5U7OICYB_wmcNfj1VVBET6ZEOELidA?e=o1tFnr).
Please unarchive and put all the files in the `data` folder.

### Usage

To train a model for dataset **d1** on multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 python d1/train_c.py --config d1/config/s0_np2048_pn.yaml
```

After training, to evaluate the result, please add a `OUTPUT_PATH` in the same config file pointing to the log folder of the trained model, and then run:
```
CUDA_VISIBLE_DEVICES=0,1 python d1/test_c.py --config d1/config/s0_np2048_pn.yaml
```

To train a model for dataset **d2** on multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 python d2/train_c.py --config d2/config/s1_np20480_r0.0_mng4096.yaml
```
After training, to evaluate the result, please add a `OUTPUT_PATH` in the same config file pointing to the log folder of the trained model, and then run:
```
CUDA_VISIBLE_DEVICES=0,1 python d2/test_c.py --config d2/config/s1_np20480_r0.0_mng4096.yaml
```

### Note
The code in this repo is under active development.

### License
Our code is released under MIT License (see LICENSE file for details).