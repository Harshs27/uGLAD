## uGLAD  
Sparse graph recovery by optimizing deep unrolled networks. This work proposes `uGLAD` (Sparse graph recovery by optimizing deep unrolled networks. [arxiv](<https://arxiv.org/abs/2205.11610>)) which is a unsupervised version of a previous `GLAD` model (GLAD: Learning Sparse Graph Recovery (ICLR 2020 - [link](<https://openreview.net/forum?id=BkxpMTEtPB>)).  

## Talk  
https://www.youtube.com/watch?v=Mx9VSQJACsA

Key benefits & features:  
- Solution to Graphical Lasso: A better alternative to solve the Graphical Lasso problem as
    - The neural networks of the uGLAD enable adaptive choices of the hyperparameters which leads to better performance than the existing algorithms  
     - No need to pre-specify the sparsity related regularization hyperparameters  
    - Requires less number of iterations to converge due to neural network based acceleration of the unrolled optimization algorithm (Alternating Minimization)    
    - GPU based acceleration can be leveraged  
    - Novel `consensus` strategy which robustly handles missing values by leveraging the multi-task learning ability of the model   
    - Multi-task learning mode that solves the graphical lasso objective to recover multiple graphs with a single `uGLAD` model  
- Glasso loss function: The loss is the logdet objective of the graphical lasso `1/M(-1*log|theta|+ <S, theta>)`, where `M=num_samples, S=input covariance matrix, theta=predicted precision matrix`.  
- Ease of usability: Matches the I/O signature of `sklearn GraphicalLassoCV`, so easy to plug-in to the existing code.  

### uGLAD architecture: Unrolled deep model

<p align="center">
  <img src="https://github.com/Harshs27/uGLAD/blob/main/.images/architecture.PNG" width="100" title="uGLAD architecture: Unrolled deep model" />
</p>

<p align="center">
  <img src="https://github.com/Harshs27/uGLAD/blob/main/.images/nn-architecture1.PNG" width="300" title="uGLAD architecture: Neural Network details" />
</p>

## Setup 
### Users

```bash
pip install uglad
```

### Developers
The `setup.sh` file contains the complete procedure of creating a conda environment to run uGLAD model. 
```bash
bash setup.sh
```

## demo-uGLAD notebook  
A minimalist working example of uGLAD is given in `examples/demo-uGLAD.ipynb` notebook. It is a good entry point to understand the code structure as well as the uGLAD model.  

## Citation
If you find this method useful, kindly cite the following 2 associated papers:

- `uGLAD`: Sparse graph recovery by optimizing deep unrolled networks. [arxiv](<https://arxiv.org/abs/2205.11610>)  
@inproceedings{  
shrivastava2022a,  
title={A deep learning approach to recover conditional independence graphs},  
author={Harsh Shrivastava and Urszula Chajewska and Robin Abraham and Xinshi Chen},  
booktitle={NeurIPS 2022 Workshop: New Frontiers in Graph Learning},  
year={2022},  
url={https://openreview.net/forum?id=kEwzoI3Am4c}  
}  

- `GLAD`:  
@article{shrivastava2019glad,  
  title={GLAD: Learning sparse graph recovery},  
  author={Shrivastava, Harsh and Chen, Xinshi and Chen, Binghong and Lan, Guanghui and Aluru, Srinvas and Liu, Han and Song, Le},  
  journal={arXiv preprint arXiv:1906.00271},  
  year={2019}  
}     
