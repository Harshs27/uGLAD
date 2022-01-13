## uGLAD  
An unsupervised deep learning model to recover sparse graphs. This work proposes `uGLAD` which is a unsupervised version of a previous `GLAD` model (GLAD: Learning Sparse Graph Recovery (ICLR 2020 - [link](<https://openreview.net/forum?id=BkxpMTEtPB>)).  

Key Benefits & features:  
- Solution to Graphical Lasso: A better alternative to solve the Graphical Lasso problem as
    - GPU based acceleration can be leveraged
    - Requires less number of iterations to converge due to neural network based acceleration of the unrolled optimization algorithm (Alternating Minimization).  
    - No need to pre-specify the sparsity related regularization hyperparameters. uGLAD models them using neural networks and are optimized for the glasso loss function.  
- Glasso loss functions: The loss is the logdet objective of the graphical lasso `1/M(-1*log|theta|+ <S, theta>)`, where `M=num_samples, S=input covariance matrix, theta=predicted precision matrix`.  
- Ease of usability: Matches the I/O signature of `sklearn GraphicalLassoCV`, so easy to plug-in to the existing code.  

## uGLAD architecture
![uGLAD architecture: Unrolled deep model](https://github.com/Harshs27/uGLAD/blob/main/.images/architecture.PNG?raw=true =20x80)  

<!-- <object data="https://github.com/Harshs27/uGLAD/blob/main/.images/architecture.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/Harshs27/uGLAD/blob/main/.images/architecture.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/Harshs27/uGLAD/blob/main/.images/architecture.pdf">Download PDF</a>.</p>
    </embed>
</object> -->

## Setup  
The `setup.sh` file contains the complete procedure of creating a conda environment to run mGLAD model. run `bash setup.sh`    
In case of dependencies conflict, one can alternatively use this command `conda env create --name uGLAD --file=environment.yml`.  

## demo-uGLAD notebook  
A minimalist working example of uGLAD. It is a good entry point to understand the code structure as well as the GLAD model.  

## Citation
If you find this method useful, kindly cite the following 2 associated papers:

- uGLAD:  

- GLAD:  
@article{shrivastava2019glad,
  title={GLAD: Learning sparse graph recovery},
  author={Shrivastava, Harsh and Chen, Xinshi and Chen, Binghong and Lan, Guanghui and Aluru, Srinvas and Liu, Han and Song, Le},
  journal={arXiv preprint arXiv:1906.00271},
  year={2019}
}
