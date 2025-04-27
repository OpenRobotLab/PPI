# INSTALLATION
## PPI Env

```bash
conda create -n ppi python=3.8
conda activate ppi

git clone https://github.com/OpenRobotLab/PPI.git
cd PPI
pip install -e .

```
### (1) PyTorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
We recommend using `CUDA 11.8`. Please install the torch version that matches your cuda version.

### (2) Third Party Packages
```bash
pip install -r requirements.txt
```

### (3) pytorch3D
```bash
conda install gxx_linux-64
mkdir repos
cd repos
git clone https://github.com/facebookresearch/pytorch3d.git pytorch3d
cd pytorch3d
pip install .
cd ..
```
Building wheels for pytorch3d may take a while.

### (4) SAM and GroundingDINO (for data preprocession)

```bash
# install sam
# cd repos
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install .
cd ..

# install groundingdino
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install .
cd ..

```

## RLBench2 Env
### (1) PyRep and Coppelia Simulator
PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git. 

```bash
# cd repos
git clone https://github.com/markusgrotz/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: modify `PATH_TO_YOUR_COPPELIASIM` to your path)

```bash
export COPPELIASIM_ROOT=PATH_TO_YOUR_COPPELIASIM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -e .
cd ..
```

## (2) Extended RLBench

Please use my [RLBench fork](https://github.com/yuyinyang3y/RLBench/). 

```bash
# cd repos
git clone https://github.com/yuyinyang3y/RLBench.git

cd RLBench
pip install -e .
cd ..
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

## (3) YARR

Please use my [YARR fork](https://github.com/yuyinyang3y/YARR/).

```bash
# cd repos
git clone https://github.com/yuyinyang3y/YARR.git 

cd YARR
pip install -e .
cd ..
```

You should be ready to go!


