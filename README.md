# LiSCa-Net
## Tongtong Wang; Lina Zhuang
[Link to paper](XXXXXXXXXXXXXX)

***
**Abstract:**
To overcome the insufficient exploitation of spectral discriminability, parameter redundancy, and limited generalization of prevailing ``spatial-priority'' zero-shot hyperspectral image (HSI) mixed-noise removal schemes, this paper proposes a Lightweight Self-Supervised Spectral-Spatial Cascade Network (LiSCa-Net). 
It constructs a two-stage architecture of ``spectral purification-spatial refinement'': stage I suppresses non-Gaussian sparse noise via a pixel-wise 1-D blind-spot network that exclusively leverages spectral correlation; stage II equalizes band-wise variance through a noise-whitening transform and removes residual Gaussian noise with a 2-D Noise2Noise module. 
Entirely trained on the test image without clean samples or noise priors, the network contains only $26.2k$ parameters and permits on-device deployment. 
Extensive experiments on simulated and real data show that LiSCa-Net achieves optimal performance in mixed noise scenarios, with excellent texture detail preservation, and has 1-2 orders of magnitude fewer parameters than other methods. 
***
## Flowchart and Network Architecture

![Flowchart and network architecture](figs/flowchart.png)

## Requirements
We tested the implementation in Python 3.10.

## Datasets
The test HSIs are available on [Google Drive](https://drive.google.com/drive/folders/1-xbTMAbYWZOrAcfL2MkyNOiW_hXPoRBq?usp=sharing). 
The folder contains clean HSIs and their corresponding noisy versions, simulated from the clean data according to the five noise cases described in the paper.

```
Put the downloaded data into the [Datasets] folder.

Alternatively, you can generate noisy HSIs according to the code.
```
## Denoising Experiments
```
cd LiSCa-Net
python run_exps_simu.py datasets.scene_name=WashingtonDC noise.case=case3 device=cuda gpu_ids=0 save_dir=./Result/
python run_exps_simu.py datasets.scene_name=pavia noise.case=case5
```


## Citation
If you find the code helpful in your resarch or work, please cite:
```
XXXXXXXXXXXXX
```

## Contact Information
Please contact me if there is any question. 
Tongtong Wang: [1822116421@qq.com](1822116421@qq.com)

Aerospace Information Research Institute

Chinese Academy of Sciences


