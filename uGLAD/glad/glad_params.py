import torch
import torch.nn as nn
from torch import Tensor


class GladParams(nn.Module):
    """GLAD model hyperparameters.

    The AM hyperparameters are parameterized in this class.
    `rho`, `lambda`, and `theta_init_offset` are learnable.
    """

    def __init__(
        self, theta_init_offset: float, nF: int, H: int, USE_CUDA: bool = False
    ) -> None:
        """Initialize the GLAD model.

        Args:
            theta_init_offset (float): Initial eigenvalue offset, set to a high value > 0.1.
            nF (int): Number of input features for entrywise thresholding.
            H (int): Hidden layer size for neural networks.
            USE_CUDA (bool, optional): If True, use GPU; otherwise, use CPU. Defaults to False.
        """
        super().__init__()
        self.dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        # Learnable parameter for theta initialization
        self.theta_init_offset = nn.Parameter(
            torch.Tensor([theta_init_offset]).type(self.dtype)
        )
        self.nF = nF  # number of input features
        self.H = H  # hidden layer size
        # Define neural networks for rho and lambda
        self.rho_l1 = self._build_rho_nn()
        self.lambda_f = self._build_lambda_nn()
        # Constant zero tensor for thresholding
        self.zero = torch.Tensor([0]).type(self.dtype)

    def _build_rho_nn(self) -> nn.Sequential:
        """Builds the rho neural network used for entrywise thresholding.

        Returns:
            nn.Sequential: A fully connected neural network with tanh and sigmoid activations.
        """
        l1 = nn.Linear(self.nF, self.H).type(self.dtype)
        lH1 = nn.Linear(self.H, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(), lH1, nn.Tanh(), l2, nn.Sigmoid()).type(
            self.dtype
        )

    def _build_lambda_nn(self) -> nn.Sequential:
        """Builds the lambda neural network.

        Returns:
            nn.Sequential: A fully connected neural network with tanh and sigmoid activations.
        """
        l1 = nn.Linear(2, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(), l2, nn.Sigmoid()).type(self.dtype)

    def eta_forward(self, X: Tensor, S: Tensor, k: int, F3: Tensor = None) -> Tensor:
        """Performs entrywise thresholding based on the learned rho values.

        Args:
            X (Tensor): Input tensor of shape (batch_size, shape1, shape2).
            S (Tensor): Second input tensor of the same shape as X.
            k (int): Iteration index (not used explicitly in this function).
            F3 (Tensor, optional): Additional feature tensor. Defaults to None.

        Returns:
            Tensor: Thresholded tensor with the same shape as X.
        """
        batch_size, shape1, shape2 = X.shape
        Xr = X.reshape(batch_size, -1, 1)
        Sr = S.reshape(batch_size, -1, 1)
        feature_vector = torch.cat((Xr, Sr), dim=-1)
        if F3 is not None:
            F3r = F3.reshape(batch_size, -1, 1)
            feature_vector = torch.cat((feature_vector, F3r), dim=-1)
        rho_val = self.rho_l1(feature_vector).reshape(X.shape)
        return torch.sign(X) * torch.max(self.zero, torch.abs(X) - rho_val)

    def lambda_forward(self, normF: float, prev_lambda: float, k: int = 0) -> Tensor:
        """Computes the lambda update using the learned neural network.

        Args:
            normF (float): Norm value of the feature matrix.
            prev_lambda (float): Previous lambda value.
            k (int, optional): Iteration index (not explicitly used). Defaults to 0.

        Returns:
            Tensor: Updated lambda value.
        """
        feature_vector = torch.Tensor([normF, prev_lambda]).type(self.dtype)
        return self.lambda_f(feature_vector)