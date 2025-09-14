# 0 download pytorch
```shell
cd ~
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.1.0
git submodule update --init --recursive
```

# 1 compiler source code
```shell
export USE_QNNPACK=0
export USE_NNPACK=0
export USE_NNPACK=off
export USE_MKLDNN=0
export USE_NNPAXCK=0
export USE_NINJA=0
export USE_PYTORCH_QNNPACK=0
export USE_PYTORCH_QNNPACK=off
export USE_ROCM=0
export USE_FBGEMM=0
export BUILD_BINARY=0
export USE_PRECOMPOLED_HEADERS=1
export USE_CUDA=1
export USE_DISTRUBUTED=1
export USE_OPENCV=0
export CMAKE_BUILD_TYPE=Debug
export MAX_JOBS=16
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
python3 setup.py bdist_wheel
# sudo CMAKE_BUILD_TYPE=Debug python3 setup.py develop
```

# 2 set environment variables
```shell
export PYTHONPATH=/root/pytorch/build/lib.linux-x86_64-3.8:$PYTHONPATH
```
