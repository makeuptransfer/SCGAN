# SCGAN
Implementation of CVPR 2021 paper "Spatially-invariant Style-codes Controlled Makeup Transfer"
## Prepare
The pre-trained model is avaiable at https://drive.google.com/file/d/1t1Hbgqqzc_rV5v3gF7HuJ-xiuEVNb8sh/view?usp=sharing.

Put the "G.pth" and "vgg.pth" in "./checkpoints" and "./" respectively.

Environments:python=3.8, pytorch=1.6.0, Ubuntu=20.04.1 LTS
## Train
Put the train-list of makeup images in "./MT-Dataset/makeup.txt" and the train-list of non-makeup images in "./MT-Dataset/non-makeup.txt"

Use the "./scripts/handle_parsing.py" to convert the origin MT-Dataset's seg labels

Use `python sc.py --phase train` to train
## Test
### 1.Global Makeup Transfer
`python sc.py --phase test`

![Global Makeup Transfer](https://github.com/makeuptransfer/SCGAN/blob/master/global_transferred.jpg)
### 2.Part-specific Makeup Transfer
`python sc.py --phase test --partial`

![Part-specific Makeup Transfer](https://github.com/makeuptransfer/SCGAN/blob/master/partial_transferred.jpg)
### 3.Global Interpolation
`python sc.py --phase test --interpolation`

![Global Interpolation](https://github.com/makeuptransfer/SCGAN/blob/master/global_interpolation_transferred.jpg)
### 4.Part-specific Interpolation
`python sc.py --phase test --partial --interpolation`

![Part-specific Interpolation](https://github.com/makeuptransfer/SCGAN/blob/master/partial_interpolation_transferred.jpg)
