# 3D stack-of-spirals MRI

This package contains Python code for acquiring 
and reconstructing 3D stack-of-spirals data.
The acquisition code (work in progress) is based on 
[PyPulseq](https://github.com/imr-framework/pypulseq),
which is a Python package for vendor-agnostic
MRI pulse sequence design.
Reconstruction is performed using [MIRTorch](https://github.com/guanhuaw/MIRTorch), 
which is a PyTorch-based image reconstruction toolbox.

# Step 0: Installation of Python environment

In this step, we install a separate Python environment
using [Conda](https://docs.conda.io/en/latest/)
(a package manager and environment system).
For our purposes, we install Miniconda, which is
a minimal installer which includes only Python and a few 
basic packages. The necessary packages for us are then installed
in the next step.

The installation instructions for Miniconda can be found at 
(https://docs.conda.io/en/main/miniconda.html). This section provides
detailed instructions for Linux, but the link above also contains
instructions for other operating systems. For Linux users,
download the suitable Miniconda installer and install it. 
For e.g.,
```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
$ bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
```

At the end of installation, a prompt asks 
```bash
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
```
to which the recommended answer is [yes](https://docs.anaconda.com/free/anaconda/reference/faq/).

Create a new conda environment to run our stack-of-spirals code. This
helps in isolating the desired packages required for `sosp3d`.
Change the name `myenv` as desired.

```bash
$ conda create -n myenv python=3.10
```

# Step 1: Install packages

Activate the newly created conda environment.
```bash
$ conda activate myenv
```
You can navigate between different conda environments using
the `conda activate` command followed by the name of the
environment. To obtain a list of available conda environments,
use the command `$ conda env list`.

## PyTorch
Install the [Pytorch](https://pytorch.org/) package using the Conda package manager. 
For e.g., Linux users can
install the latest version of PyTorch (with CUDA capabilities) using
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
This installs all required dependencies and could potentially involve 
a considerable download size (~2.71GB on our Linux machine). The packages
installed within a given conda environment can be obtained using the command 
`$ conda list` after activating the desired environment.
This is a good way to confirm that the necessary packages have indeed
been installed.

## MIRTorch
[MIRTorch](https://github.com/guanhuaw/MIRTorch) is a PyTorch-based toolbox 
for image reconstruction. Clone the MIRTorch repository
(https://github.com/guanhuaw/MIRTorch)
to a suitable location
locally and navigate to the local directory. Run the setup.py file in the MIRTorch folder.
```bash
cd /path/to/MIRTorch
python setup.py install
```

To double-check that the packages are installed correctly, run the following import
statements in Python, and they should run without any errors. Change `myenv` to the 
name of your conda environment.

```bash
$ conda activate myenv
$ python
>>> import torch
>>> from mirtorch.linear import Gmri
```









