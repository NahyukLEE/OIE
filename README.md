# Outdoor-Illumination-Estimation

A research project on outdoor illumination estimating methods.

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

## Action Items
Nahyuk will provide some action items for each week after weekly meetings with Prof. Hong.
#### Week0 (May 20 FRI ~ May 22 SUN)
- Nahyuk: Dataset Creation with [NYC 3D Model](https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-nyc-3d-model-download.page) using Blender.
  - Create both 360-images-dataset and fov-images-dataset and compare estimation models(VGG/ResNet)' results. (with Deadline ~Week1)
- Sejin: Review some papers which have quite similar objecttive to ours.
  - _"Deep Outdoor Illumination Estimation", CVPR 2017_, https://arxiv.org/abs/1611.06403 (with Deadline ~Week0)
  - _"All-Weather Deep Outdoor Lighting Estimation", CVPR 2019_, https://arxiv.org/abs/1906.04909 (with Deadline ~Week1)
#### Week1 (May 22 MON ~ May 29 SUN)
- TBU
