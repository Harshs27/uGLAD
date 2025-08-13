import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, Optimizer

from uglad.glad.torch_sqrtm import MatrixSquareRoot

torch_sqrtm = MatrixSquareRoot.apply


def get_optimizers(
    model_glad: nn.Module, lr_glad: float = 0.002, use_optimizer: str = "adam"
) -> Optimizer:
    """Creates and returns an optimizer for the given model.

    Args:
        model_glad (nn.Module): The GLAD model whose parameters will be optimized.
        lr_glad (float, optional): Learning rate for the optimizer. Defaults to 0.002.
        use_optimizer (str, optional): Name of the optimizer to use. Currently supports 'adam'.
            Defaults to 'adam'.

    Returns:
        Optimizer: The initialized optimizer.

    Raises:
        ValueError: If an unsupported optimizer type is provided.
    """
    if use_optimizer == "adam":
        return Adam(
            model_glad.parameters(),
            lr=lr_glad,
            betas=(0.9, 0.999),
            eps=1e-08,
            # weight_decay=0
        )
    raise ValueError("Optimizer not found! Supported optimizers: ['adam']")


def batch_matrix_sqrt(A: Tensor) -> Tensor:
    """Computes the matrix square root for a single or batch of PSD matrices.

    Args:
        A (Tensor): A positive semi-definite (PSD) matrix or a batch of PSD matrices.
            - Shape (n, m, m) for a batch of n matrices.
            - Shape (m, m) for a single matrix.

    Returns:
        Tensor: The matrix square root of A with the same shape as the input.
    """
    # A should be PSD
    if A.dim() == 2:  # Single matrix case
        return torch_sqrtm(A)
    n = A.shape[0]  # Number of matrices in batch
    sqrtm_torch = torch.zeros(A.shape).type_as(A)  # Initialize output tensor
    for i in range(n):
        sqrtm_torch[i] = torch_sqrtm(A[i])  # Compute sqrt for each matrix
    return sqrtm_torch


def get_frobenius_norm(A: Tensor, single: bool = False) -> Tensor:
    """Computes the Frobenius norm squared of a single or batch of matrices.

    Args:
        A (Tensor): A single matrix (m, n) or a batch of matrices (b, m, n).
        single (bool): If True, computes the norm for a single matrix. Otherwise,
                    computes the mean Frobenius norm for a batch.

    Returns:
        Tensor: Frobenius norm squared if `single=True`, else mean Frobenius norm squared.
    """
    return torch.sum(A**2) if single else torch.mean(torch.sum(A**2, dim=(1, 2)))


def glad(
    Sb: Tensor,
    model,
    lambda_init: float = 1,
    L: int = 15,
    INIT_DIAG: int = 0,
    USE_CUDA: bool = False,
) -> tuple[Tensor, Tensor]:
    """Unrolling the Alternating Minimization algorithm which takes in the
    sample covariance (batch mode), runs the iterations of the AM updates and
    returns the precision matrix. The hyperparameters are modeled as small
    neural networks which are to be learned from the backprop signal of the
    loss function.

        Args:
        Sb (Tensor): Covariance matrix of shape (batch, dim, dim).
        model: GLAD neural network parameters (theta_init, rho, lambda).
        lambda_init (float): Initial lambda value.
        L (int): Number of unrolled AM iterations.
        INIT_DIAG (int): Initialization of initial theta as:
            - 0: (S + theta_init_offset * I)^-1
            - 1: (diag(S) + theta_init_offset * I)^-1
        USE_CUDA (bool): Use GPU if True, otherwise use CPU.

    Returns:
        tuple[Tensor, Tensor]:
            - theta_pred (Tensor): Estimated precision matrix (batch, dim, dim).
            - loss (Tensor): Graphical lasso objective function value.
    """
    D = Sb.shape[-1]  # Dimension of matrix
    if Sb.dim() == 2:
        Sb = Sb.reshape(1, Sb.shape[0], Sb.shape[1])  # Ensure batch dimension
    # Initializing the theta based on INIT_DIAG mode
    if INIT_DIAG == 1:
        # print('extract batchwise diagonals, add offset and take inverse')
        batch_diags = 1 / (
            torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset
        )
        theta_init = torch.diag_embed(batch_diags)
    else:
        # print('(S+theta_offset*I)^-1 is used')
        theta_init = torch.inverse(
            Sb + model.theta_init_offset * torch.eye(D).expand_as(Sb).type_as(Sb)
        )
    theta_pred = theta_init
    identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
    # diagonal mask
    #    mask = torch.eye(Sb.shape[-1], Sb.shape[-1]).byte()
    #    dim = Sb.shape[-1]
    #    mask1 = torch.ones(dim, dim) - torch.eye(dim, dim)
    if USE_CUDA is True:
        identity_mat = identity_mat.cuda()
    #        mask = mask.cuda()
    #        mask1 = mask1.cuda()

    zero = torch.Tensor([0])
    dtype = torch.FloatTensor
    if USE_CUDA is True:
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    lambda_k = model.lambda_forward(zero + lambda_init, zero, k=0)
    for k in range(L):
        # GLAD CELL
        # Compute b and its transformation for matrix square root
        b = 1.0 / lambda_k * Sb - theta_pred
        b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0 / lambda_k * identity_mat
        sqrt_term = batch_matrix_sqrt(b2_4ac)
        theta_k1 = 1.0 / 2 * (-1 * b + sqrt_term)
        # Update theta using GLAD's eta_forward function
        theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred)
        # update the lambda
        lambda_k = model.lambda_forward(
            torch.Tensor([get_frobenius_norm(theta_pred - theta_k1)]).type(dtype),
            lambda_k,
            k,
        )
    return theta_pred