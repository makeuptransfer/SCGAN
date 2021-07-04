# SCGAN
## Prepare
The pre-trained model is avaiable at https://drive.google.com/file/d/1t1Hbgqqzc_rV5v3gF7HuJ-xiuEVNb8sh/view?usp=sharing.

Put the "G.pth" and "vgg.pth" in "./checkpoints" and "./" respectively.

Environments:python=3.8, pytorch=1.6.0, Ubuntu=20.04.1 LTS
## Test
### 1.Global Makeup Transfer
`python test.py`

![Global Makeup Transfer](https://github.com/makeuptransfer/SCGAN/blob/master/global_transferred.jpg)
### 2.Part-specific Makeup Transfer
`python test.py --partial`

![Part-specific Makeup Transfer](https://github.com/makeuptransfer/SCGAN/blob/master/partial_transferred.jpg)
### 3.Global Interpolation
`python test.py --interpolation`

![Global Interpolation](https://github.com/makeuptransfer/SCGAN/blob/master/global_interpolation_transferred.jpg)
### 4.Part-specific Interpolation
`python test.py --partial --interpolation`

![Part-specific Interpolation](https://github.com/makeuptransfer/SCGAN/blob/master/partial_interpolation_transferred.jpg)
