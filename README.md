# OIEID: Outdoor-Illumination-Estimation from Intrinsic Decomposition

A research project(for my B.S. Thesis) on outdoor illumination estimating methods.

## Installation
This repository will be developed and tested on **Ubuntu 18.04 LTS**.
- Detail environments will be updated(CUDA, Python, PyTorch etc.)

### Environment Setup
- TBU: I will provide our setup script, soon.

(Optional) One may need to modify CUDA path and Anaconda path in the "setup.sh" script. \
One can setup the environment by running the script as follows:
```
bash setup.sh oie
conda activate oie
```

## Collaboration
### Experiment Management
We use [WandB](https://wandb.ai/site) to manage multiple experiments on different machines with various users.
If you don't have an account, please make one.
After preparing the account, please let Nahyuk know your username to invite you into the WandB Team, "cau_cvar_oie".
If everything goes perfect, you may find out a WandB project named "cau_cvar_oie/oie".
### Contributing
For contributing, we recommend you to make your own branch and request the Nahyuk to merge it.
Do not commit your modification on the main branch. \
For example, if Nahyuk wants to implement {MODULE}, he can do it as follows:
```
git checkout -b nahyuklee/{MODULE}
# Do some commits on the "nahyuklee/{MODULE}" branch.
# Then, open a pull request to merge the branch into the main branch.
```
## Dataset Creation
You can see detail instructions for dataset creation(using Blender) in this [MD](./docs/DATA.md).
