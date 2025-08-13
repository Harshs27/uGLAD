"""
The main file to train/test the uGLAD algorithm.
Contains code to generate data, run training and the
loss function.
"""
import copy
import io
import pickle
import sys
from time import time
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pyvis import network as net
from scipy.linalg import solve
from scipy.stats import multivariate_normal
from sklearn import covariance
from sklearn.model_selection import KFold

import uglad.utils.prepare_data as prepare_data
from uglad.glad import glad

# Helper functions for uGLAD
from uglad.glad.glad_params import GladParams
from uglad.utils.metrics import report_metrics_all


# Wrapper class to match sklearn package #############
class uGLAD_GL(object):
    def __init__(self):
        """Wrapper class to match the sklearn GraphicalLassoCV output signature.
        Initializes and fits the uGLAD model for graphical lasso with the given data.
        """
        super(uGLAD_GL, self).__init__()
        self.covariance_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.location_: Optional[np.ndarray] = None  # mean vector
        self.model_glad: Optional[object] = None

    def fit(
        self,
        X: np.ndarray,
        true_theta: Optional[np.ndarray] = None,
        eval_offset: float = 0.1,
        centered: bool = False,
        epochs: int = 250,
        lr: float = 0.002,
        INIT_DIAG: int = 0,
        L: int = 15,
        verbose: bool = True,
        k_fold: int = 3,
        mode: str = "direct",
        node_names: Optional[list] = None,
    ) -> dict[str, float]:
        """Takes in the samples X and returns
        a uGLAD model which stores the corresponding
        covariance and precision matrices.

        Args:
            X (np.ndarray): Input data of shape (num_samples, num_features).
            true_theta (np.ndarray, optional): True precision matrix of shape (D, D).
            eval_offset (float, optional): Adjustment for ill-conditioned covariance.
            centered (bool, optional): Whether to mean-center the data before fitting.
            epochs (int, optional): Number of epochs for training.
            lr (float, optional): Learning rate for the Adam optimizer.
            INIT_DIAG (int, optional): Initialization strategy for GLAD (0 or 1).
            L (int, optional): Number of unrolled iterations of GLAD.
            verbose (bool, optional): Whether to print verbose training output.
            k_fold (int, optional): Number of folds for cross-validation.
            mode (str, optional): Mode of operation ('direct', 'cv', or 'missing').
            node_names (list, optional): List of node names.

        Returns:
            Dict[str, float]: Dictionary of comparison metrics between predicted and
                            true precision matrix.
        """
        print("Running uGLAD")
        start = time()
        print("Processing the input table for basic compatibility check")
        X = prepare_data.process_table(pd.DataFrame(X), NORM="min_max", VERBOSE=verbose)
        X = np.array(X)
        # Running the uGLAD model
        M, D = X.shape
        # Reshaping due to GLAD algorithm requirements
        Xb = X.reshape(1, M, D)
        true_theta_b = None
        if true_theta is not None:
            true_theta_b = true_theta.reshape(1, D, D)
        if mode == "missing":
            print("Handling missing data")
            pred_theta, compare_theta, model_glad = run_uGLAD_missing(
                Xb,
                trueTheta=true_theta_b,
                eval_offset=eval_offset,
                EPOCHS=epochs,
                lr=lr,
                INIT_DIAG=INIT_DIAG,
                L=L,
                VERBOSE=verbose,
                K_batch=k_fold,
            )
        elif mode == "cv" and k_fold >= 0:
            print(f"CV mode: {k_fold}-fold")
            pred_theta, compare_theta, model_glad = run_uGLAD_CV(
                Xb,
                trueTheta=true_theta_b,
                eval_offset=eval_offset,
                EPOCHS=epochs,
                lr=lr,
                INIT_DIAG=INIT_DIAG,
                L=L,
                VERBOSE=verbose,
                k_fold=k_fold,
            )
        elif mode == "direct":
            print("Direct Mode")
            pred_theta, compare_theta, model_glad = run_uGLAD_direct(
                Xb,
                trueTheta=true_theta_b,
                eval_offset=eval_offset,
                EPOCHS=epochs,
                lr=lr,
                INIT_DIAG=INIT_DIAG,
                L=L,
                VERBOSE=verbose,
            )
        else:
            print(
                f"ERROR Please enter K-fold value in valid range [0, ), currently \
                entered {k_fold}; Check mode {mode}"
            )
            sys.exit(0)
        # np.dot((X-mu)T, (X-mu)) / X.shape[0]
        self.covariance_ = covariance.empirical_covariance(X, assume_centered=centered)
        self.location_ = X.mean(axis=0)
        # If node names then assign to the object
        if node_names is not None:
            self.node_names_ = node_names
        else:
            self.node_names_ = [f"node_{i}" for i in range(D)]
        if pred_theta is not None:
            self.precision_ = pred_theta[0].detach().numpy()
        if model_glad is not None:
            self.model_glad = model_glad
        print(f"Total runtime: {time()-start} secs\n")
        return compare_theta


# redefine the function to include type hints and docstrings
class uGLAD_multitask(object):
    def __init__(self):
        """Initializing the uGLAD model in multi-task
        mode. It saves the covariance and predicted
        precision matrices for the input batch of data
        """
        super(uGLAD_multitask, self).__init__()
        self.covariance_: list[np.ndarray] = []
        self.precision_: np.ndarray | None = None
        self.model_glad: object | None = None

    def fit(
        self,
        Xb: list[np.ndarray],
        true_theta_b: np.ndarray | None = None,
        eval_offset: float = 0.1,
        centered: bool = False,
        epochs: int = 250,
        lr: float = 0.002,
        INIT_DIAG: int = 0,
        L: int = 15,
        verbose: bool = True,
    ) -> list[dict]:
        """Takes in the samples X and returns a uGLAD model which stores the
        corresponding covariance and precision matrices.

        Args:
            Xb (list[np.ndarray]): Input data of shape (num_samples, num_features).
            true_theta_b (np.ndarray, optional): True precision matrix of shape (D, D).
            eval_offset (float, optional): Adjustment for ill-conditioned covariance.
            centered (bool, optional): Whether to mean-center the data before fitting.
            epochs (int, optional): Number of epochs for training.
            lr (float, optional): Learning rate for the Adam optimizer.
            INIT_DIAG (int, optional): Initialization strategy for GLAD (0 or 1).
            L (int, optional): Number of unrolled iterations of GLAD.
            verbose (bool, optional): Whether to print verbose training output.
        Returns:
            compare_theta (list[dict]): List of dictionaries of comparison metrics
                between predicted and true precision matrix.
        """
        print("Running uGLAD in multi-task mode")
        start = time()
        print("Processing the input table for basic compatibility check")
        processed_Xb = []
        for X in Xb:
            X = prepare_data.process_table(
                pd.DataFrame(X), NORM="min_max", VERBOSE=verbose
            )
            processed_Xb.append(np.array(X))
        Xb = processed_Xb
        # Running the uGLAD model
        pred_theta, compare_theta, model_glad = run_uGLAD_multitask(
            Xb,
            trueTheta=true_theta_b,
            eval_offset=eval_offset,
            EPOCHS=epochs,
            lr=lr,
            INIT_DIAG=INIT_DIAG,
            L=L,
            VERBOSE=verbose,
        )
        # np.dot((X-mu)T, (X-mu)) / X.shape[0]
        self.covariance_ = []
        for b in range(len(Xb)):
            self.covariance_.append(
                covariance.empirical_covariance(Xb[b], assume_centered=centered)
            )
        self.covariance_ = np.array(self.covariance_)
        self.precision_ = pred_theta.detach().numpy()
        self.model_glad = model_glad
        print(f"Total runtime: {time()-start} secs\n")
        return compare_theta


#####################################################################


# Functions to prepare model ######################
def init_uGLAD(
    lr: float, theta_init_offset: float = 1.0, nF: int = 3, H: int = 3
) -> tuple[GladParams, object]:
    """Initialize the uGLAD model and optimizer.

    Args:
        lr (float): Learning rate for the Adam optimizer.
        theta_init_offset (float): Offset for initializing theta.
        nF (int): Number of filters.
        H (int): Number of hidden layers.
    Returns:
        model (GladParams): Initialized uGLAD model.
        optimizer (object): Adam optimizer for the model.
    """
    model = GladParams(theta_init_offset=theta_init_offset, nF=nF, H=H)
    optimizer = glad.get_optimizers(model, lr_glad=lr)
    return model, optimizer


def forward_uGLAD(
    Sb: torch.Tensor,
    model_glad: GladParams,
    L: int = 15,
    INIT_DIAG: int = 0,
    loss_Sb: Optional[torch.Tensor] = None,
    struct_theta: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for the uGLAD model.
    Run the input through the unsupervised GLAD algorithm.
    It executes the following steps in batch mode
    1. Run the GLAD model to get predicted precision matrix
    2. Calculate the glasso-loss

    Adding the case where the precision matrix structure is provided

    Args:
        Sb (torch.Tensor): Covariance matrix.
        model_glad (GladParams): uGLAD model.
        L (int): Number of unrolled iterations of GLAD.
        INIT_DIAG (int): Initialization strategy for GLAD (0 or 1).
        loss_Sb (torch.Tensor, optional): Loss covariance matrix.
        struct_theta (torch.Tensor, optional): Structural prior.
    Returns:
        predTheta (torch.Tensor): Predicted precision matrix.
        loss (torch.Tensor): Glasso loss.
    """
    # 1. Running the GLAD model
    predTheta = glad.glad(Sb, model_glad, L=L, INIT_DIAG=INIT_DIAG)
    # 2. Calculate the glasso-loss
    if loss_Sb is None:
        loss = loss_uGLAD(predTheta, Sb, struct_theta=struct_theta)
    else:
        loss = loss_uGLAD(predTheta, loss_Sb, struct_theta=struct_theta)
    return predTheta, loss


def loss_uGLAD(
    theta: torch.Tensor, S: torch.Tensor, struct_theta: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate the glasso loss for the uGLAD model.
    The objective function of the graphical lasso which is
    the loss function for the unsupervised learning of glad
    loss-glasso = 1/M(-log|theta| + <S, theta>)

    NOTE: We fix the batch size B=1 for `uGLAD`

    Args:
        theta (torch.Tensor): Predicted precision matrix.
        S (torch.Tensor): Covariance matrix.
        struct_theta (torch.Tensor, optional): Structural prior.
    Returns:
        glasso_loss (torch.Tensor): Glasso loss.
    """
    B, D, _ = S.shape
    t1 = -1 * torch.logdet(theta)
    # Batch Matrix multiplication: torch.bmm
    t21 = torch.einsum("bij, bjk -> bik", S, theta)
    # getting the trace (batch mode)
    t2 = torch.einsum("jii->j", t21)
    # print(t1, torch.det(theta), t2)
    # regularization term
    # tr = 1e-02 * torch.sum(torch.abs(theta))
    glasso_loss = torch.sum(t1 + t2) / B  # sum over the batch
    # # Adding the structure loss if provided
    # if struct_theta is not None:
    #     # struct_theta is a [0,1] matrix, use its complement, make sure that diagonal is 0
    #     # struct_loss is entrywise multiplication of the complement and theta
    #     struct_loss = torch.sum(torch.abs(theta*(1-(struct_theta+torch.eye(D)))))/B
    #     # print the value of the glasso and struct loss
    #     print(f'Glasso loss: {glasso_loss} Struct loss: {struct_loss}')
    #     glasso_loss += struct_loss
    # Optional Structural Loss
    if struct_theta is not None:
        # Create identity matrix and expand for batch processing
        eye_mask = torch.eye(D, device=theta.device).expand(B, -1, -1)
        mask = (1 - struct_theta) - eye_mask  # Ensures diagonal is zero
        # absolute loss
        # struct_loss = torch.sum(torch.abs(theta * mask)) / B  # Compute masked loss
        # log-cosh loss (smooth approximation of L1)
        struct_loss = torch.sum(torch.log(torch.cosh(theta * mask))) / B
        print(f"Glasso loss: {glasso_loss}  Struct loss: {struct_loss}")
        glasso_loss += struct_loss  # Add structure loss
    return glasso_loss


def run_uGLAD_direct(
    Xb: np.ndarray,
    trueTheta: Optional[np.ndarray] = None,
    eval_offset: float = 0.1,
    EPOCHS: int = 250,
    lr: float = 0.002,
    INIT_DIAG: int = 0,
    L: int = 15,
    VERBOSE: bool = True,
) -> tuple[torch.Tensor, Optional[dict], GladParams]:
    """Running the uGLAD algorithm in direct mode. This function
    takes in the input sample matrix and the true precision
    matrix (if available) and runs the uGLAD algorithm
    to learn the precision matrix. It also reports the
    comparison metrics if the true precision matrix is provided.

    Args:
        Xb (np.array 1xMxD): The input sample matrix
        trueTheta (np.array 1xDxD): The corresponding
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initialization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out
    Returns:
        predTheta (torch.Tensor 1xDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    # Calculating the batch covariance
    Sb = prepare_data.get_covariance(Xb, offset=eval_offset)  # BxDxD
    # Converting the data to torch
    Xb = prepare_data.convert_to_torch(Xb, req_grad=False)
    Sb = prepare_data.convert_to_torch(Sb, req_grad=False)
    if trueTheta is not None:
        trueTheta = prepare_data.convert_to_torch(trueTheta, req_grad=False)
    B, M, D = Xb.shape
    # NOTE: We fix the batch size B=1 for `uGLAD`
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(
        lr=lr, theta_init_offset=1.0, nF=3, H=3  # eval_offset, #1.0,
    )
    print("Initialized parameters for uGLAD theta_init_offset=1.0,nF=3,H=3")

    PRINT_EVERY = int(EPOCHS / 10)  # print max 10 times per training
    loss_values = []  # Track loss values
    # Optimizing for the glasso loss
    for e in range(EPOCHS):
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss
        predTheta, loss = forward_uGLAD(
            Sb,
            model_glad,
            L=L,
            INIT_DIAG=INIT_DIAG,
            struct_theta=trueTheta,
        )
        # Check for NaN loss
        if torch.isnan(loss):
            print(
                f"Warning: NaN loss encountered at epoch {e}.\
                Try updating the parameters and train."
            )
            break
        # calculate the backward gradients
        loss.backward()
        if not e % PRINT_EVERY and VERBOSE:
            print(f"epoch:{e}/{EPOCHS} loss:{loss.detach().numpy()}")
        # updating the optimizer params with the grads
        optimizer_glad.step()
        # Store loss value
        loss_values.append(loss.detach().numpy())
    # Plot loss curve after training
    plot_loss_curve(loss_values)
    # reporting the metrics if true thetas provided
    compare_theta = None
    if trueTheta is not None:
        for b in range(B):
            compare_theta = report_metrics_all(
                trueTheta[b].detach().numpy(), predTheta[b].detach().numpy()
            )
            print(f"Compare - {compare_theta}")
    return predTheta, compare_theta, model_glad


def run_uGLAD_CV(
    Xb: np.ndarray,
    trueTheta: Optional[np.ndarray] = None,
    eval_offset: float = 0.1,
    EPOCHS: int = 250,
    lr: float = 0.002,
    INIT_DIAG: int = 0,
    L: int = 15,
    VERBOSE: bool = True,
    k_fold: int = 5,
) -> tuple[torch.Tensor, Optional[dict], GladParams]:
    """Running the uGLAD algorithm in cross-validation mode. This function
    takes in the input sample matrix and the true precision
    matrix (if available) and runs the uGLAD algorithm
    to learn the precision matrix. It also reports the
    comparison metrics if the true precision matrix is provided.
    The model is trained in k-fold cross-validation mode.

    Args:
        Xb (np.array 1xMxD): The input sample matrix
        trueTheta (np.array 1xDxD): The corresponding
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initialization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out
        k_fold (int): Number of folds for cross-validation
    Returns:
        predTheta (torch.Tensor 1xDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    # Batch size is fixed to 1
    Sb = prepare_data.get_covariance(Xb, offset=eval_offset)
    Sb = prepare_data.convert_to_torch(Sb, req_grad=False)
    # Splitting into k-fold for cross-validation
    kf = KFold(n_splits=k_fold)
    # For each fold, collect the best model and the glasso-loss value
    results_Kfold = {}
    for _k, (train, test) in enumerate(kf.split(Xb[0])):
        if VERBOSE:
            print(f"Fold num {_k}")
        Xb_train = np.expand_dims(Xb[0][train], axis=0)  # 1 x Mtrain x D
        Xb_test = np.expand_dims(Xb[0][test], axis=0)  # 1 x Mtest x D
        # Calculating the batch covariance
        Sb_train = prepare_data.get_covariance(Xb_train, offset=eval_offset)  # BxDxD
        Sb_test = prepare_data.get_covariance(Xb_test, offset=eval_offset)  # BxDxD
        # Converting the data to torch
        Sb_train = prepare_data.convert_to_torch(Sb_train, req_grad=False)
        Sb_test = prepare_data.convert_to_torch(Sb_test, req_grad=False)
        if trueTheta is not None:
            trueTheta = prepare_data.convert_to_torch(trueTheta, req_grad=False)
        B, M, D = Xb_train.shape
        # NOTE: We fix the batch size B=1 for `uGLAD'
        # model and optimizer for uGLAD
        model_glad, optimizer_glad = init_uGLAD(lr=lr, theta_init_offset=1.0, nF=3, H=3)
        # Optimizing for the glasso loss
        best_test_loss = np.inf
        PRINT_EVERY = int(EPOCHS / 10)
        # print max 10 times per training
        for e in range(EPOCHS):
            # reset the grads to zero
            optimizer_glad.zero_grad()
            # calculate the loss for test and precision matrix for train
            predTheta, loss_train = forward_uGLAD(
                Sb_train, model_glad, L=L, INIT_DIAG=INIT_DIAG
            )
            with torch.no_grad():
                _, loss_test = forward_uGLAD(
                    Sb_test, model_glad, L=L, INIT_DIAG=INIT_DIAG
                )
            # calculate the backward gradients
            loss_train.backward()
            # updating the optimizer params with the grads
            optimizer_glad.step()
            # Printing output
            _loss = loss_test.detach().numpy()
            if not e % PRINT_EVERY and VERBOSE:
                print(f"Fold {_k}: epoch:{e}/{EPOCHS} test-loss:{_loss}")
            # Updating the best model for this fold
            if _loss < best_test_loss:  # and e%10==9:
                if VERBOSE and not e % PRINT_EVERY:
                    print(
                        f"Fold {_k}: epoch:{e}/{EPOCHS}: Updating the best model \
                        with test-loss {_loss}"
                    )
                best_model_kfold = copy.deepcopy(model_glad)
                best_test_loss = _loss
        # updating with the best model and loss for the current fold
        results_Kfold[_k] = {}
        results_Kfold[_k]["test_loss"] = best_test_loss
        results_Kfold[_k]["model"] = best_model_kfold
        if VERBOSE:
            print("\n")

    # Strategy I: Select the best model from the results Kfold dictionary
    # with the best score on the test fold.
    # print(f'Using Strategy I to select the best model')
    best_loss = np.inf
    for _k in results_Kfold.keys():
        curr_loss = results_Kfold[_k]["test_loss"]
        if curr_loss < best_loss:
            model_glad = results_Kfold[_k]["model"]
            best_loss = curr_loss

    # Run the best model on the complete data to retrieve the
    # final predTheta (precision matrix)
    with torch.no_grad():
        predTheta, total_loss = forward_uGLAD(Sb, model_glad, L=L, INIT_DIAG=INIT_DIAG)

    # reporting the metrics if true theta is provided
    compare_theta = None
    if trueTheta is not None:
        for b in range(B):
            compare_theta = report_metrics_all(
                trueTheta[b].detach().numpy(), predTheta[b].detach().numpy()
            )
        print(f"Comparison - {compare_theta}")
    return predTheta, compare_theta, model_glad


def run_uGLAD_missing(
    Xb: np.ndarray,
    trueTheta: Optional[np.ndarray] = None,
    eval_offset: float = 0.1,
    EPOCHS: int = 250,
    lr: float = 0.002,
    INIT_DIAG: int = 0,
    L: int = 15,
    VERBOSE: bool = True,
    K_batch: int = 3,
) -> tuple[torch.Tensor, Optional[dict], GladParams]:
    """Running the uGLAD algorithm in missing data mode.We do a
    row-subsample of the input data and then train using multi-task
    learning approach to obtain the final precision matrix. This function
    takes in the input sample matrix and the true precision
    matrix (if available) and runs the uGLAD algorithm
    to learn the precision matrix. It also reports the
    comparison metrics if the true precision matrix is provided.
    The model is trained in missing data mode.

    Args:
        Xb (np.array 1xMxD): The input sample matrix
        trueTheta (np.array 1xDxD): The corresponding
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initialization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out
        K_batch (int): Number of batches for missing data
    Returns:
        predTheta (torch.Tensor 1xDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    # Batch size is fixed to 1
    if K_batch == 0:
        K_batch = 3  # setting the default
    # Step I:  Do statistical mean imputation
    Xb = mean_imputation(Xb)
    # Step II: Getting the batches and preparing data for uGLAD
    Sb = prepare_data.get_covariance(Xb, offset=eval_offset)
    Sb = prepare_data.convert_to_torch(Sb, req_grad=False)
    # Splitting into k-fold for getting row-subsampled batches
    kf = KFold(n_splits=K_batch)
    print(f"Creating K={K_batch} row-subsampled batches")
    # Collect all the subsample in batch form (list): K x M' x D
    X_K = [Xb[0][Idx] for Idx, _ in kf.split(Xb[0])]
    # Calculating the batch covariance
    S_K = prepare_data.get_covariance(X_K, offset=eval_offset)  # BxDxD
    # Converting the data to torch
    S_K = prepare_data.convert_to_torch(S_K, req_grad=False)
    # Initialize the model and prepare theta if provided
    if trueTheta is not None:
        trueTheta = prepare_data.convert_to_torch(trueTheta, req_grad=False)
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(lr=lr, theta_init_offset=1.0, nF=3, H=3)
    # STEP III: Optimizing for the glasso loss
    PRINT_EVERY = int(EPOCHS / 10)
    # print max 10 times per training
    for e in range(EPOCHS):
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss and precision matrix
        predTheta, loss = forward_uGLAD(
            S_K, model_glad, L=L, INIT_DIAG=INIT_DIAG, loss_Sb=Sb
        )
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer_glad.step()
        # Printing output
        _loss = loss.detach().numpy()
        if not e % PRINT_EVERY and VERBOSE:
            print(f"epoch:{e}/{EPOCHS} loss:{_loss}")

    # STEP IV: Getting the final precision matrix
    print("Getting the final precision matrix using the consensus strategy")
    predTheta = get_final_precision_from_batch(predTheta, type="min")

    # reporting the metrics if true theta is provided
    compare_theta = None
    if trueTheta is not None:
        compare_theta = report_metrics_all(
            trueTheta[0].detach().numpy(), predTheta[0].detach().numpy()
        )
        print(f"Comparison - {compare_theta}")

    return predTheta, compare_theta, model_glad


def mean_imputation(Xb: np.ndarray) -> np.ndarray:
    """Mean imputation for missing data. This function
    takes in the input sample matrix and does mean
    imputation for the missing values. It also checks
    if any column is full of NaNs and raises an error.
    Args:
        Xb (np.array 1xMxD): The input sample matrix
    Returns:
        Xb (np.array 1xMxD): The input sample matrix
    """
    Xb = Xb[0]
    # Mean of columns (ignoring NaNs)
    col_mean = np.nanmean(Xb, axis=0)
    # Find indices that you need to replace
    inds = np.where(np.isnan(Xb))
    # Place column means in the indices. Align the arrays using take
    Xb[inds] = np.take(col_mean, inds[1])
    # Check if any column is full of NaNs, raise sys.exit()
    if np.isnan(np.sum(Xb)):
        print("ERROR: One or more columns have all NaNs")
        sys.exit(0)
    # Reshaping Xb with an extra dimension for compatability with glad
    Xb = np.expand_dims(Xb, axis=0)
    return Xb


def get_final_precision_from_batch(
    predTheta: torch.Tensor, type: str = "min"
) -> torch.Tensor:
    """Get the final precision matrix from the batch
    of precision matrices. This function takes in the
    batch of precision matrices and returns the final
    precision matrix using the consensus strategy.
    It does the following:
    1. Get the value term using min/mean
    2. Get the sign term using max_count_sign
    3. Get the final precision matrix using value_term and sign_term

    The predTheta contains a batch of K precision
    matrices. This function calculates the final
    precision matrix by following the consensus
    strategy

    Theta^{f}_{i,j} = max-count(sign(Theta^K_{i,j}))
                        * min/mean{|Theta^K_{i,j}|}
    (`min` is the recommended setting)

    Args:
        predTheta (torch.Tensor): Batch of precision matrices.
        type (str): Type of consensus strategy ('min' or 'mean').
    Returns:
        predTheta (torch.Tensor): Final precision matrix.
    """
    K, _, D = predTheta.shape
    # get the value term
    if type == "min":
        value_term = torch.min(torch.abs(predTheta), 0)[0]
    elif type == "mean":
        value_term = torch.mean(torch.abs(predTheta), 0)[0]
    else:
        print(f"Enter valid type min/mean, currently {type}")
        sys.exit(0)
    # get the sign term
    max_count_sign = torch.sum(torch.sign(predTheta), 0)
    # If sign is 0, then assign +1
    max_count_sign[max_count_sign >= 0] = 1
    max_count_sign[max_count_sign < 0] = -1
    # Get the final precision matrix
    predTheta = max_count_sign * value_term
    return predTheta.reshape(1, D, D)


def run_uGLAD_multitask(
    Xb: list[np.ndarray],
    trueTheta: Optional[np.ndarray] = None,
    eval_offset: float = 0.1,
    EPOCHS: int = 250,
    lr: float = 0.002,
    INIT_DIAG: int = 0,
    L: int = 15,
    VERBOSE: bool = True,
) -> tuple[torch.Tensor, Optional[dict], GladParams]:
    """Running the uGLAD algorithm in multi-task mode. We
    train using multi-task learning approach to obtain
    the final precision matrices for the batch of input data.
    It also reports the comparison metrics if the true precision
    matrix is provided. The model is trained in multi-task mode.

    Args:
        Xb (list of 2D np.array):  The input sample matrix K * [M' x D]
            NOTE: num_samples can be different for different data
        trueTheta (np.array KxDxD): The corresponding
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initialization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out

    Returns:
        predTheta (torch.Tensor BxDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    K = len(Xb)
    # Getting the batches and preparing data for uGLAD
    Sb = prepare_data.get_covariance(Xb, offset=eval_offset)
    Sb = prepare_data.convert_to_torch(Sb, req_grad=False)
    # Initialize the model and prepare theta if provided
    if trueTheta is not None:
        trueTheta = prepare_data.convert_to_torch(trueTheta, req_grad=False)
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(lr=lr, theta_init_offset=1.0, nF=3, H=3)
    # Optimizing for the glasso loss
    PRINT_EVERY = int(EPOCHS / 10)
    # print max 10 times per training
    for e in range(EPOCHS):
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss and precision matrix
        predTheta, loss = forward_uGLAD(Sb, model_glad, L=L, INIT_DIAG=INIT_DIAG)
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer_glad.step()
        # Printing output
        _loss = loss.detach().numpy()
        if not e % PRINT_EVERY and VERBOSE:
            print(f"epoch:{e}/{EPOCHS} loss:{_loss}")

    # reporting the metrics if true theta is provided
    compare_theta = []
    if trueTheta is not None:
        for b in range(K):
            rM = report_metrics_all(
                trueTheta[b].detach().numpy(), predTheta[b].detach().numpy()
            )
            print(f"Metrics for graph {b}: {rM}\n")
            compare_theta.append(rM)
    return predTheta, compare_theta, model_glad


#################################################################
# Functions to get partial correlation matrix and visualization


def get_partial_correlations(precision: np.ndarray) -> np.ndarray:
    """Get the partial correlation matrix from the
    precision matrix. It applies the following

    Formula: rho_ij = -p_ij/sqrt(p_ii * p_jj)

    Args:
        precision (2D np.array): The precision matrix

    Returns:
        rho (2D np.array): The partial correlations
    """
    precision = np.array(precision)
    D = precision.shape[0]
    rho = np.zeros((D, D))
    for i in range(D):  # rows
        for j in range(D):  # columns
            if i == j:  # diagonal elements
                rho[i][j] = 1
            elif j < i:  # symmetric
                rho[i][j] = rho[j][i]
            else:  # i > j
                num = -1 * precision[i][j]
                den = np.sqrt(precision[i][i] * precision[j][j])
                rho[i][j] = num / den
    return rho


# Plot the graph
def graph_from_partial_correlations(
    rho: np.ndarray,
    names: list[str],  # node names
    sparsity: float = 1,
    title: str = "",
    fig_size: int = 12,
    PLOT: bool = True,
    save_file: Optional[str] = None,
    roundOFF: int = 5,
) -> tuple[nx.Graph, Optional[bytes], list[str]]:
    """Create a graph from the partial correlation matrix.

    Args:
        rho (np.ndarray): Partial correlation matrix.
        names (list[str]): Node names.
        sparsity (float, optional): Sparsity level. Defaults to 1.
        title (str, optional): Title of the plot. Defaults
            to ''.
        fig_size (int, optional): Figure size. Defaults to 12.
        PLOT (bool, optional): Whether to plot the graph.
            Defaults to True.
        save_file (str, optional): Path to save the plot.
            Defaults to None.
        roundOFF (int, optional): Number of decimal places to
            round the edge weights. Defaults to 5.
    Returns:
        tuple[nx.Graph, Optional[bytes], list[str]]:
            The graph object, image bytes, and edge list.
    """
    G = nx.Graph()
    G.add_nodes_from(names)
    D = rho.shape[-1]

    # determining the threshold to maintain the sparsity level of the graph
    def upper_tri_indexing(A):
        m = A.shape[0]
        r, c = np.triu_indices(m, 1)
        return A[r, c]

    rho_upper = upper_tri_indexing(np.abs(rho))
    num_non_zeros = int(sparsity * len(rho_upper))
    rho_upper.sort()
    th = rho_upper[-num_non_zeros]
    # print(f'Sparsity {sparsity} using threshold {th}')
    th_pos, th_neg = th, -1 * th

    graph_edge_list = []
    for i in range(D):
        for j in range(i + 1, D):
            if rho[i, j] > th_pos:
                G.add_edge(
                    names[i],
                    names[j],
                    color="green",
                    weight=round(rho[i, j], roundOFF),
                    label="+",
                )
                _edge = (
                    "("
                    + names[i]
                    + ", "
                    + names[j]
                    + ", "
                    + str(round(rho[i, j], roundOFF))
                    + ", green)"
                )
                graph_edge_list.append(_edge)
            elif rho[i, j] < th_neg:
                G.add_edge(
                    names[i],
                    names[j],
                    color="red",
                    weight=round(rho[i, j], roundOFF),
                    label="-",
                )
                _edge = (
                    "("
                    + names[i]
                    + ", "
                    + names[j]
                    + ", "
                    + str(round(rho[i, j], roundOFF))
                    + ", red)"
                )
                graph_edge_list.append(_edge)

    # if PLOT: print(f'graph edges {graph_edge_list, len(graph_edge_list)}')

    edge_colors = [G.edges[e]["color"] for e in G.edges]
    edge_width = np.array([abs(G.edges[e]["weight"]) for e in G.edges])
    # Scaling the intensity of the edge_weights for viewing purposes
    if len(edge_width) > 0:
        edge_width = edge_width / np.max(np.abs(edge_width))
    image_bytes = None
    if PLOT:
        fig = plt.figure(1, figsize=(fig_size, fig_size))
        plt.title(title)
        n_edges = len(G.edges())
        pos = nx.spring_layout(G, scale=0.2, k=1 / np.sqrt(n_edges + 10))
        # pos = nx.nx_agraph.graphviz_layout(G, prog='fdp') #'fdp', 'sfdp', 'neato'
        nx.draw_networkx_nodes(G, pos, node_color="grey", node_size=100)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_width)
        y_off = 0.008
        nx.draw_networkx_labels(
            G, pos={k: ([v[0], v[1] + y_off]) for k, v in pos.items()}
        )
        plt.title(f"{title}", fontsize=20)
        plt.margins(0.15)
        plt.tight_layout()
        # saving the file
        if save_file:
            plt.savefig(save_file, bbox_inches="tight")
        # Saving the figure in-memory
        buf = io.BytesIO()
        plt.savefig(buf)
        # getting the image in bytes
        buf.seek(0)
        image_bytes = buf.getvalue()  # Image.open(buf, mode='r')
        buf.close()
        # closing the plt
        plt.close(fig)
    return G, image_bytes, graph_edge_list


def get_interactive_graph(
    G: nx.Graph, title: str = "", node_PREFIX: Optional[str] = "ObsVal"
) -> net.Network:
    """
    Create an interactive graph using Pyvis from a NetworkX graph.

    Args:
        G (nx.Graph): The input graph.
        title (str, optional): Title of the graph. Defaults to ''.
        node_PREFIX (str, optional): Prefix for node labels.
            Defaults to 'ObsVal'.

    Returns:
        net.Network: The interactive Pyvis network graph.
    """
    # Initialize Pyvis network object
    # NOTE: notebook=True is required for Jupyter Notebook
    # but not needed for other environments
    Gv = net.Network(
        notebook=True,
        height="750px",
        width="100%",
        #     bgcolor='#222222', font_color='white',
        heading=title,
    )
    Gv.from_nx(G.copy(), show_edge_weights=True, edge_weight_transf=(lambda x: x))
    for e in Gv.edges:
        e["title"] = str(e["width"])
        e["value"] = abs(e["width"])
    if node_PREFIX is not None:
        for n in Gv.nodes:
            n["title"] = node_PREFIX + ":" + n["category"]
    Gv.show_buttons()
    return Gv


def viz_graph_from_precision(
    theta: np.ndarray, feature_names: list[str], sparsity: float = 0.1, title: str = ""
) -> tuple[nx.Graph, net.Network]:
    """
    Visualizing the CI graph from the precision matrix.

    Args:
        theta (np.ndarray): The precision matrix.
        feature_names (list[str]): List of feature names.
        sparsity (float, optional): Sparsity level. Defaults to 0.1.
        title (str, optional): Title of the plot. Defaults to ''.

    Returns:
        tuple[nx.Graph, net.Network]: The graph object and interactive network.
    """
    rho = get_partial_correlations(theta)
    Gr, _, _ = graph_from_partial_correlations(rho, feature_names, sparsity=sparsity)
    print(f"Num nodes: {len(Gr.nodes)}")
    Gv = get_interactive_graph(Gr, title, node_PREFIX=None)
    # visualize using Gv.show('viz.html')
    return Gr, Gv


def darken_color(color: str, factor: float = 0.3) -> str:
    """
    Darkens an input color by scaling its RGB components.

    Args:
        color (str): The original color in a hex or named format.
        factor (float, optional): Scaling factor for darkening (0-1).
            Defaults to 0.3.

    Returns:
        str: The darkened color in hex format.
    """
    rgb = mcolors.to_rgb(color)  # Convert color to RGB tuple (0-1 range)
    dark_rgb = tuple(max(0, c * factor) for c in rgb)  # Scale each component
    return mcolors.to_hex(dark_rgb)  # Convert back to hex


def build_interactive_graph(
    G: nx.Graph,
    title: str = "",
    node_sizes: Optional[dict[str, int]] = None,
    default_node_size: int = 30,
    viz_file: Optional[str] = "viz.html",
) -> net.Network:
    """
    Builds an interactive graph using Pyvis with customized node sizes and colors based on
    node types.

    Args:
        G (nx.Graph): The input graph.
        title (str, optional): Title of the graph. Defaults to ''.
        node_sizes (dict[str, int], optional): Dictionary mapping node names to sizes.
            If None, all nodes will have a default size of `default_node_size`.
            If provided, node sizes will be scaled by `default_node_size`. Defaults to None.
        default_node_size (int, optional): Default size for nodes. If `node_sizes` is given,
            values are multiplied by this factor. Defaults to 30.
        viz_file (str, optional): The file name for saving the visualization.
            Defaults to "viz.html".

    Returns:
        net.Network: The interactive Pyvis network graph.
    """
    # Initialize Pyvis network object
    Gv = net.Network(notebook=True, height="750px", width="100%", heading=title)
    # Load the NetworkX graph into Pyvis
    Gv.from_nx(G.copy(), show_edge_weights=True, edge_weight_transf=(lambda x: x))
    # Set edge properties for display
    for e in Gv.edges:
        e["title"] = e["display_relation"] + "::" + e["edge_source"]
        e["value"] = abs(e["width"])  # Control edge thickness
    # Set node sizes; if not provided, set a default size for all nodes
    if node_sizes is None:
        node_sizes = {str(n["id"]): default_node_size for n in Gv.nodes}
        node_scores = {str(n["id"]): 1 for n in Gv.nodes}
    else:
        node_scores = node_sizes.copy()
        # Scale provided node sizes by the default size
        node_sizes = {
            str(k): (1 + v) * default_node_size for k, v in node_sizes.items()
        }

    # Get the unique node types from node attributes and generate a color map
    node_types = set(
        nx.get_node_attributes(G, "node_type").values()
    )  # Unique node types
    # color_map = {
    # node_type: color[4:] for node_type, color in
    #   zip(node_types, mcolors.TABLEAU_COLORS)
    # }
    base_colors = list(mcolors.TABLEAU_COLORS.values())  # Get Tableau colors
    dark_colors = [darken_color(c, factor=0.7) for c in base_colors]  # Darken colors
    # assert len(node_types) == len(
    #     dark_colors
    # ), "Mismatch in lengths of node_types and dark_colors"
    color_map = {
        node_type: color
        for node_type, color in zip(node_types, dark_colors, strict=False)
    }
    # Assign node sizes and colors based on node types
    for n in Gv.nodes:
        node_id = str(n["id"])
        n["label"] = G.nodes[node_id].get(
            "node_name", node_id
        )  # Default to node ID if missing
        # Set node size from the provided or default sizes
        n["size"] = node_sizes.get(node_id, default_node_size)
        # Display node size in hover tooltip
        n["title"] = f"{{score={node_scores.get(node_id, 1):.3f}}}"
        # Set node color based on node type, defaulting to 'gray' if not found
        node_type = G.nodes[node_id].get("node_type", "default")  # Default to 'default'
        # Assign color based on node type
        n["color"] = color_map.get(node_type, "gray")

    # Enable interactive controls (zoom, pan, etc.)
    Gv.show_buttons()
    # Save the visualization to an HTML file
    Gv.show(viz_file)

    # Add a legend to the saved HTML file
    legend_html = "<div style='position: fixed; bottom: 20px; left: 20px; background: white;\
        padding: 10px; border-radius: 5px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);'>"
    legend_html += "<strong>Legend:</strong><br>"
    for node_type, color in color_map.items():
        legend_html += f"<div style='display: flex;\
            align-items: center;'><div style='width: 15px; height: 15px; background: {color};\
            margin-right: 5px;'></div>{node_type}</div>"
    legend_html += "</div>"

    # Read the generated HTML and insert the legend
    with open(viz_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Insert legend before closing </body> tag
    html_content = html_content.replace("</body>", legend_html + "</body>")

    # Save the updated HTML file
    with open(viz_file, "w", encoding="utf-8") as file:
        file.write(html_content)
    return Gv


######################################################################


def save_uGLAD_model(obj: uGLAD_GL, filepath: str) -> None:
    """Saves the uGLAD_GL object using pickle, handling the torch model separately.

    Args:
        obj: Instance of uGLAD_GL to be saved.
        filepath: Path to save the object.
    """
    # Temporarily extract the PyTorch model
    model_path = filepath + "_model.pt"
    if obj.model_glad is not None:
        torch.save(obj.model_glad.state_dict(), model_path)
    # Remove the model temporarily
    obj_dict = obj.__dict__.copy()
    obj_dict["model_glad"] = None  # Avoid saving torch model in pickle
    # Save the remaining object using pickle
    with open(filepath, "wb") as f:
        pickle.dump(obj_dict, f)
    print(f"Model saved successfully at {filepath} and {model_path}")


def load_uGLAD_model(filepath: str) -> uGLAD_GL:
    """Loads the uGLAD_GL object, restoring the torch model separately.

    Args:
        filepath: Path where the object was saved.

    Returns:
        obj: Loaded uGLAD_GL instance.
    """
    model_path = filepath + "_model.pt"
    # Load the saved object (excluding the model)
    with open(filepath, "rb") as f:
        obj_dict = pickle.load(f)
    # Create a new instance and restore attributes
    obj = uGLAD_GL()
    obj.__dict__.update(obj_dict)
    # Restore the PyTorch model if it exists
    if obj_dict["model_glad"] is not None:
        obj.model_glad = torch.load(model_path)
        print("Torch model restored successfully.")
    print(f"Model loaded successfully from {filepath}")
    return obj


def conditional_gaussian_with_probabilities(
    precision: np.ndarray,
    mean: np.ndarray,
    observed_idx: list[int],
    observed_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the conditional mean, covariance, and MAP probabilities for a
    Gaussian distribution given observed variables.

    Args:
        precision (np.ndarray):
            Precision matrix (inverse of covariance), shape (n, n).
        mean (np.ndarray):
            Mean vector of the Gaussian, shape (n,).
        observed_idx (List[int]):
            List of indices corresponding to observed variables.
        observed_values (np.ndarray):
            Observed values at the given indices, shape (len(observed_idx),).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - Full mean vector with conditional estimates (shape (n,)).
            - Conditional covariance matrix (shape (n_unobserved, n_unobserved)).
            - Probability density function (PDF) value at the MAP estimate.
    """
    # Total number of variables in the model
    n: int = len(mean)
    # Determine unobserved indices
    unobserved_idx: list[int] = list(set(range(n)) - set(observed_idx))
    # Partition precision matrix
    Lambda_11: np.ndarray = precision[np.ix_(unobserved_idx, unobserved_idx)]
    Lambda_12: np.ndarray = precision[np.ix_(unobserved_idx, observed_idx)]
    # Extract corresponding means
    mean_unobs: np.ndarray = mean[unobserved_idx]
    mean_obs: np.ndarray = mean[observed_idx]
    # Compute conditional mean (MAP estimate)
    conditional_mean: np.ndarray = mean_unobs - solve(
        Lambda_11, Lambda_12 @ (observed_values - mean_obs)
    )
    # Compute conditional covariance matrix
    conditional_cov: np.ndarray = np.linalg.inv(Lambda_11)
    # Construct full mean vector
    full_mean: np.ndarray = np.zeros(n)
    full_mean[unobserved_idx] = conditional_mean
    full_mean[observed_idx] = observed_values
    # Compute probability density function (PDF) at MAP estimate
    pdf: float = multivariate_normal.pdf(
        conditional_mean, mean=conditional_mean, cov=conditional_cov
    )
    return full_mean, conditional_cov, pdf


def compute_map_estimate(
    observed_nodes: dict[str, float], model_uGLAD: uGLAD_GL
) -> np.ndarray:
    """
    Computes the Maximum A Posteriori (MAP) estimate for a Gaussian prior and
    Gaussian likelihood, given observed node values.

    Args:
        observed_nodes (Dict[str, float]):
            A dictionary mapping observed node names to their values.
        model_uGLAD (uG.uGLAD_GL):
            The trained uGLAD model containing the precision matrix, mean vector,
            and node names.

    Returns:
        np.ndarray:
            The MAP estimate of the mean parameter, clamped between 0 and 1.
    """
    # Extract model parameters
    precision_matrix: np.ndarray = model_uGLAD.precision_  # Precision matrix
    mean_vector: np.ndarray = model_uGLAD.location_  # Mean vector
    node_names: list[str] = model_uGLAD.node_names_  # List of node names
    # Get observed indices and values
    observed_idx: list[int] = [node_names.index(node) for node in observed_nodes.keys()]
    observed_values: list[float] = list(observed_nodes.values())
    print(f"Observed Indices: {observed_idx}, Observed Values: {observed_values}")
    # Compute the MAP estimate using the conditional Gaussian function
    cond_mean, _, _ = conditional_gaussian_with_probabilities(
        precision_matrix, mean_vector, observed_idx, observed_values
    )
    # Clamp values between 0 and 1 to ensure valid probabilities
    return np.clip(cond_mean, 0, 1)


def compute_graph_statistics(G: nx.Graph) -> pd.DataFrame:
    """
    Computes various statistics for a given NetworkX graph and returns them in a DataFrame.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        pd.DataFrame: A DataFrame containing the following statistics:
            - num_nodes: Number of nodes in the graph.
            - num_edges: Number of edges in the graph.
            - avg_degree: Average node degree.
            - density: Graph density.
            - avg_clustering: Average clustering coefficient.
            - transitivity: Global transitivity (triangle density).
            - diameter: Diameter of the graph (if connected).
            - avg_shortest_path: Average shortest path length (if connected).
    """
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
        if G.number_of_nodes() > 0
        else 0,
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }
    # Compute diameter and average shortest path only if the graph is connected
    if nx.is_connected(G):
        stats["diameter"] = nx.diameter(G)
        stats["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        stats["diameter"] = None  # Not defined for disconnected graphs
        stats["avg_shortest_path"] = None
    # Convert the statistics dictionary to a pandas DataFrame
    stats_df = pd.DataFrame([stats])
    return stats_df


def plot_loss_curve(
    loss_values: list[float], title: str = "Training Loss Curve"
) -> None:
    """
    Plots the loss curve over epochs.

    Args:
        loss_values (list[float]): List of loss values over training epochs.
        title (str, optional): Title of the plot. Defaults to "Training Loss Curve".
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_values)), loss_values, label="Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()


######################################################################
# DO NOT USE
def post_threshold(theta: np.ndarray, s: float = 80.0) -> np.ndarray:
    """Apply post-hoc thresholding to zero out the
    entries based on the input sparsity percentile.
    Usually we take conservative value of sparsity
    percentage, so that we do not miss important
    edges.

    Args:
        theta (2d np array): The DxD precision matrix
        s (float): Percentile sparsity desired

    Returns:
        theta (2d np array): The DxD precision matrix
    """
    # getting the threshold for s percentile
    cutoff = np.percentile(np.abs(theta), s)
    theta[np.abs(theta) < cutoff] = 0
    return theta