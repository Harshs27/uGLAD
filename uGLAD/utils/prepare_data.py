import networkx as nx
import numpy as np
from sklearn import covariance
import torch

def convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(dtype)
    data.requires_grad = req_grad
    return data


def eigVal_conditionNum(A):
    """Calculates the eigenvalues and the condition 
    number of the input matrix A

    condition number = max(|eig|)/min(|eig|)
    """
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / min(np.abs(eig))
    return eig, condition_number


def getCovariance(Xb, offset=0.1):
    """Calculate the batch covariance matrix 

    Args:
        Xb (3D np array): The input sample matrices (B x M x D)
        offset (float): The eigenvalue offset in case of bad 
                        condition number
    Returns:
        Sb (3D np array): Covariance matrices (B x D x D)
    """
    Sb = []
    for X in Xb:
        S = covariance.empirical_covariance(X, assume_centered=False)
        # calculate the eigenvalue of the covariance S
        eig, con = eigVal_conditionNum(S)
        if min(eig)<=1e-3:
            # adjust the eigenvalue
            print(f'Adjust the eval: min {min(eig)}, con {con}')
            S += np.eye(S.shape[-1]) * (offset-min(eig))
            eig, con = eigVal_conditionNum(S)
            print(f'new eval: min {min(eig)}, con {con}')
        Sb.append(S)
    return np.array(Sb)


def generateRandomGraph(num_nodes, sparsity, seed=None):
    """Generate a random erdos-renyi graph with a given
    sparsity. 

    Args:
        num_nodes (int): The number of nodes in the DAG
        sparsity (float): = #edges-present/#total-edges
        seed (int, optional): set the numpy random seed

    Returns:
        edge_connections (2D np array (float)): Adj matrix
    """
    if seed: np.random.seed(seed)
    G = nx.generators.random_graphs.gnp_random_graph(
        num_nodes, 
        sparsity, 
        seed=seed, 
        directed=False
    )
    edge_connections = nx.adjacency_matrix(G).todense()
    return edge_connections


def simulateGaussianSamples(
    num_nodes,
    edge_connections, 
    num_samples, 
    seed=None, 
    u=0.1,
    w_min=0.5,
    w_max=1.0, 
    ): 
    """Simulating num_samples from a Gaussian distribution. The 
    precision matrix of the Gaussian is determined using the 
    edge_connections

    Args:
        num_nodes (int): The number of nodes in the DAG
        edge_connections (2D np array (float)): Adj matrix
        num_sample (int): The number of samples
        seed (int, optional): set the numpy random seed
        u (float): Min eigenvalue offset for the precision matrix
        w_min (float): Precision matrix entries ~Unif[w_min, w_max]
        w_max (float):  Precision matrix entries ~Unif[w_min, w_max]

    Returns:
        X (2D np array (float)): num_samples x num_nodes
        precision_mat (2D np array (float)): num_nodes x num_nodes
    """
    # zero mean of Gaussian distribution
    mean_value = 0 
    mean_normal = np.ones(num_nodes) * mean_value
    # Setting the random seed
    if seed: np.random.seed(seed)
    # uniform entry matrix [w_min, w_max]
    U = np.matrix(np.random.random((num_nodes, num_nodes))
                  * (w_max - w_min) + w_min)
    theta = np.multiply(edge_connections, U)
    # making it symmetric
    theta = (theta + theta.T)/2 + np.eye(num_nodes)
    smallest_eigval = np.min(np.linalg.eigvals(theta))
    # Just in case : to avoid numerical error in case an 
    # epsilon complex component present
    smallest_eigval = smallest_eigval.real
    # making the min eigenvalue as u
    precision_mat = theta + np.eye(num_nodes)*(u - smallest_eigval)
    # print(f'Smallest eval: {np.min(np.linalg.eigvals(precision_mat))}')
    # getting the covariance matrix (avoid the use of pinv) 
    cov = np.linalg.inv(precision_mat) 
    # get the samples 
    if seed: np.random.seed(seed)
    # Sampling data from multivariate normal distribution
    data = np.random.multivariate_normal(
        mean=mean_normal,
        cov=cov, 
        size=num_samples
        )
    return data, precision_mat  # MxD, DxD