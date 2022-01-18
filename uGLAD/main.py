"""
The main file to train/test the uGLAD algorithm.
Contains code to generate data, run training and the 
loss function.
"""
import numpy as np
from sklearn import covariance
import torch

# Helper functions for uGLAD
from uGLAD.glad.glad_params import glad_params
from uGLAD.glad import glad
from uGLAD.utils.metrics import reportMetrics

import uGLAD.utils.prepare_data as prepare_data

############### Wrapper class to match sklearn package #############
class uGLAD_GL(object):
    def __init__(self):
        """Wrapper class to match the sklearn GraphicalLassoCV
        output signature. Initializing the uGLAD model.
        """
        super(uGLAD_GL, self).__init__()
        self.covariance_ = None
        self.precision_ = None
        
    def fit(
        self, 
        X, 
        true_theta=None, 
        centered=False, 
        epochs=250, 
        verbose=True
        ):
        """Takes in the samples X and returns
        a uGLAD model which stores the corresponding
        covariance and precision matrices.
        
        Args:
            X (2D np array): num_samples x dimension
            true_theta (2D np array): dim x dim of the 
                true precision matrix
            centered (bool): Whether samples are mean 
                adjusted or not. True/False
            epochs (int): Training epochs
            verbose (bool): Print training output
        """
        X = np.array(X)
        # Running the uGLAD model
        M, D = X.shape
        # Reshaping due to GLAD algorithm requirements
        Xb = X.reshape(1, M, D)
        true_theta_b = None
        if true_theta is not None:
            true_theta_b = true_theta.reshape(1, D, D)
        pred_theta = run_uGLAD(
            Xb,
            trueTheta=true_theta_b,
            EPOCHS=epochs, 
            VERBOSE=verbose
            )
        # np.dot((X-mu)T, (X-mu)) / X.shape[0]
        self.covariance_ = covariance.empirical_covariance(
            X,
            assume_centered=centered
            )
        self.precision_ = pred_theta[0].detach().numpy()

#####################################################################

#################### Functions to generate data #####################
def get_data(
    num_nodes,
    sparsity,
    num_samples,
    batch_size=1,
    # typeG='RANDOM', 
    w_min=0.5, 
    w_max=1.0,
    eig_offset=0.1, 
    ):
    """Prepare true adj matrices as theta and then sample from 
    Gaussian to get the corresponding samples.
    
    Args:
        num_nodes (int): The number of nodes in DAG
        num_edges (int): The number of desired edges in DAG
        num_samples (int): The number of samples to simulate from DAG
        batch_size (int, optional): The number of batches
        typeG (str): RANDOM/GRID/CHAIN
        w_min (float): Precision matrix entries ~Unif[w_min, w_max]
        w_max (float):  Precision matrix entries ~Unif[w_min, w_max]
    
    Returns:
        Xb (torch.Tensor BxMxD): The sample data
        trueTheta (torch.Tensor BxDxD): The true precision matrices
    """
    Xb, trueTheta = [], []
    for b in range(batch_size):
        # I - Getting the true edge connections
        edge_connections = prepare_data.generateRandomGraph(
            num_nodes, 
            sparsity,
            #typeG=typeG
            )
        # II - Gettings samples from fitting a Gaussian distribution
        # sample the entry of the matrix 
        
        X, true_theta = prepare_data.simulateGaussianSamples(
            num_nodes,
            edge_connections,
            num_samples, 
            u=eig_offset,
            w_min=w_min,
            w_max=w_max
            )
        # collect the batch data
        Xb.append(X)
        trueTheta.append(true_theta)
    return np.array(Xb), np.array(trueTheta)
######################################################################


#################### Functions to prepare model ######################
def init_uGLAD(theta_init_offset=1.0, nF=3, H=3):
    """Initialize the GLAD model parameters and the optimizer
    to be used.

    Args:
        theta_init_offset (float): Initialization diagonal offset 
            for the pred theta (adjust eigenvalue)
        nF (int): #input features for the entrywise thresholding
        H (int): The hidden layer size to be used for the NNs
    
    Returns:
        model: class object
        optimizer: class object
    """
    model = glad_params(
        theta_init_offset=theta_init_offset,
        nF=nF, 
        H=H
        )
    optimizer = glad.get_optimizers(model)
    return model, optimizer


def forward_uGLAD(Sb, model_glad):
    """Run the input through the unsupervised GLAD algorithm.
    It executes the following steps in batch mode
    1. Run the GLAD model to get initial good regularization
    2. Calculate the glasso-loss
    
    Args:
        Sb (torch.Tensor BxDxD): The input covariance matrix
        uGLADmodel (dict): Contains the learnable params
    
    Returns:
        predTheta (torch.Tensor BxDxD): The predicted theta
        loss (torch.scalar): The glasso loss 
    """
    # 1. Running the GLAD model 
    predTheta = glad.glad(Sb, model_glad)
    # 2. Calculate the glasso-loss
    loss = loss_uGLAD(predTheta, Sb)
    return predTheta, loss


def loss_uGLAD(theta, S):
    """The objective function of the graphical lasso which is 
    the loss function for the unsupervised learning of glad
    loss-glasso = 1/M(-log|theta| + <S, theta>)

    NOTE: We fix the batch size B=1 for `uGLAD`

    Args:
        theta (tensor 3D): precision matrix BxDxD
        S (tensor 3D): covariance matrix BxDxD (dim=D)
    
    Returns:
        loss (tensor 1D): the loss value of the obj function
    """
    B, D, _ = S.shape
    t1 = -1*torch.logdet(theta)
    # Batch Matrix multiplication: torch.bmm
    t21 = torch.einsum("bij, bjk -> bik", S, theta)
    # getting the trace (batch mode)
    t2 = torch.einsum('jii->j', t21)
    # print(t1, torch.det(theta), t2) 
    # regularization term 
    # tr = 1e-02 * torch.sum(torch.abs(theta))
    glasso_loss = torch.sum(t1+t2)/B # sum over the batch
    return glasso_loss 


def run_uGLAD(Xb, trueTheta=None, eval_offset=0.1, EPOCHS=250, VERBOSE=True):
    """Running the uGLAD algorithm.
    
    Args:
        Xb (torch.Tensor BxMxD): The input sample matrix
        trueTheta (torch.Tensor BxDxD): The corresponding 
            true graphs for reporting metrics
        eval_offset (float): eigenvalue offset for 
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        VERBOSE (bool): if True, prints to sys.out

    Returns:
        predTheta (torch.Tensor BxDxD): Predicted graphs
    """
    # Calculating the batch covariance
    Sb = prepare_data.getCovariance(Xb, offset=eval_offset) # BxDxD
    # Converting the data to torch 
    Xb = prepare_data.convertToTorch(Xb, req_grad=False)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    if trueTheta is not None:
        trueTheta = prepare_data.convertToTorch(
            trueTheta,
            req_grad=False
            )
    B, M, D = Xb.shape
    # NOTE: We fix the batch size B=1 for `uGLAD`
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(
        theta_init_offset=1.0,
        nF=3,
        H=3
        )
    # Optimizing for the glasso loss
    for e in range(EPOCHS):      
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss
        predTheta, loss = forward_uGLAD(Sb, model_glad)
        # calculate the backward gradients
        loss.backward()
        if not e%25: print(f'epoch:{e}/{EPOCHS} loss:{loss.detach().numpy()}')
        # updating the optimizer params with the grads
        optimizer_glad.step()
#         print('theta_init_offset: ', model_glad.theta_init_offset)
        # reporting the metrics if true thetas provided
        if trueTheta is not None and (e+1)%EPOCHS == 0 and VERBOSE:
            for b in range(B):
                compare_theta = reportMetrics(
                    trueTheta[b].detach().numpy(), 
                    predTheta[b].detach().numpy()
                )
                print(f'Compare - {compare_theta}')
    return predTheta
######################################################################
