## SpikeDHS

This repository contains part of the code for our NeurIPS 2022 paper `Differentiable hierarchical and surrogate gradient search for spiking neural networks` 


### Environment
1. Python 3.8.*
2. CUDA 10.0
3. PyTorch 
4. TorchVision 
5. fitlog

### Install
Create a  virtual environment and activate it.
```shell
conda create -n SpikeDHS python=3.8
conda activate SpikeDHS
```
The code has been tested with PyTorch 1.6 and Cuda 10.2.
```shell
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
conda install matplotlib path.py tqdm
conda install tensorboard tensorboardX
conda install scipy scikit-image opencv
```

Install Nvidia Apex

Follow the instructions [here](https://github.com/NVIDIA/apex#quick-start). Apex is required for mixed precision training. 
Please do not use pip install apex - this will not install the correct package.

### Dataset
To evaluate/retrain our SpikeDHS network, you will need to download the CIFAR10 dataset.

### Retrain
We only provide retrain code now. You can evaluate/retrain base on our searched architecture.

Searched Architecture:
```
network_path_fea = [0,0,1,1,1,2,2,2]
cell_arch_fea = [[1, 1],
                    [0, 1],
                    [3, 2],
                    [2, 1],
                    [7, 1],
                    [8, 1]]
```

```shell
bash train.sh
```

### Acknowledgements
This repository makes liberal use of code from LEAStereo and AutoDeeplab


