from time import time
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, pearsonr
from sklearn import covariance


# Functions to generate data #####################
def generate_random_graph(
    num_nodes: int, sparsity: tuple[float, float], seed: Optional[int] = None
) -> np.ndarray:
    """Generate a random Erdos-Renyi graph with a given sparsity.

    Args:
        num_nodes (int): The number of nodes in the graph
        sparsity (Tuple[float, float]): The [min, max] probability of edges
        seed (Optional[int], optional): Set the numpy random seed (default is None)

    Returns:
        np.ndarray: Adjacency matrix (2D numpy array) of shape (num_nodes, num_nodes)
    """
    # if seed: np.random.seed(seed)
    min_s, max_s = sparsity  # Unpack sparsity values
    s = np.random.uniform(min_s, max_s, 1)[0]  # Sample the sparsity value
    # Generate the random graph using the Erdos-Renyi model
    G = nx.generators.random_graphs.gnp_random_graph(
        num_nodes, s, seed=seed, directed=False
    )
    # Convert the graph's adjacency matrix to a dense numpy array
    edge_connections = nx.adjacency_matrix(G).todense()
    return edge_connections


def simulate_gaussian_samples(
    num_nodes: int,
    edge_connections: np.ndarray,
    num_samples: int,
    seed: Optional[int] = None,
    u: float = 0.1,
    w_min: float = 0.5,
    w_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate samples from a Gaussian distribution. The precision matrix of the
    Gaussian is determined using the edge_connections.

    Args:
        num_nodes (int): The number of nodes in the DAG
        edge_connections (np.ndarray): Adjacency matrix of shape (num_nodes, num_nodes)
        num_samples (int): The number of samples to generate
        seed (Optional[int], optional): Set the numpy random seed (default is None)
        u (float, optional): Min eigenvalue offset for the precision matrix (default is 0.1)
        w_min (float, optional): Min value for precision matrix entries (default is 0.5)
        w_max (float, optional): Max value for precision matrix entries (default is 1.0)

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Generated samples of shape (num_samples, num_nodes)
            - precision_mat (np.ndarray): Precision matrix of shape (num_nodes, num_nodes)
    """
    # zero mean of Gaussian distribution
    mean_value = 0
    mean_normal = np.ones(num_nodes) * mean_value
    # Setting the random seed
    if seed:
        np.random.seed(seed)
    # uniform entry matrix [w_min, w_max]
    U = np.matrix(np.random.random((num_nodes, num_nodes)) * (w_max - w_min) + w_min)
    theta = np.multiply(edge_connections, U)
    # making it symmetric
    theta = (theta + theta.T) / 2 + np.eye(num_nodes)
    smallest_eigval = np.min(np.linalg.eigvals(theta))
    # Just in case : to avoid numerical error in case an
    # epsilon complex component present
    smallest_eigval = smallest_eigval.real
    # making the min eigenvalue as u
    precision_mat = theta + np.eye(num_nodes) * (u - smallest_eigval)
    # print(f'Smallest eval: {np.min(np.linalg.eigvals(precision_mat))}')
    # getting the covariance matrix (avoid the use of pinv)
    cov = np.linalg.inv(precision_mat)
    # get the samples
    if seed:
        np.random.seed(seed)
    # Sampling data from multivariate normal distribution
    data = np.random.multivariate_normal(mean=mean_normal, cov=cov, size=num_samples)
    # Returns the samples and the precision matrix
    return data, precision_mat  # MxD, DxD


def get_data(
    num_nodes: int,
    sparsity: tuple[float, float],
    num_samples: int,
    batch_size: int = 1,
    w_min: float = 0.5,
    w_max: float = 1.0,
    eig_offset: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare true adj matrices as theta and then sample from Gaussian to get the
    corresponding samples.

    Args:
        num_nodes (int): The number of nodes in graph
        sparsity (Tuple[float, float]): The [min, max] probability of edges
        num_samples (int): The number of samples to simulate
        batch_size (int, optional): The number of batches to generate (default 1)
        w_min (float, optional): Minimum precision matrix entry (default 0.5)
        w_max (float, optional): Maximum precision matrix entry (default 1.0)
        eig_offset (float, optional): Eigenvalue offset for the precision matrix (default 0.1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Xb (np.ndarray): Sample data of shape (batch_size, num_samples, num_nodes)
            - trueTheta (np.ndarray): True precision matrices of shape
                (batch_size, num_nodes, num_nodes)
    """
    Xb, trueTheta = [], []
    for _b in range(batch_size):
        # I - Getting the true edge connections for the graph
        edge_connections = generate_random_graph(
            num_nodes,
            sparsity,
            # typeG=typeG
        )
        # II - Simulating Gaussian samples based on edge connections
        X, true_theta = simulate_gaussian_samples(
            num_nodes,
            edge_connections,
            num_samples,
            u=eig_offset,
            w_min=w_min,
            w_max=w_max,
        )
        # collect the batch data
        Xb.append(X)
        trueTheta.append(true_theta)
    return np.array(Xb), np.array(trueTheta)


def add_noise_dropout(Xb: np.ndarray, dropout: float = 0.25) -> np.ndarray:
    """Add dropout noise to the input data by replacing a percentage of values with NaNs.

    Args:
        Xb (np.ndarray): The sample data with shape (batch_size, num_samples, num_nodes)
        dropout (float, optional): The percentage of values to replace with NaNs (default 0.25)

    Returns:
        np.ndarray: The sample data with dropout noise, with NaNs replacing a percentage
                    of values (same shape as Xb).
    """
    B, M, D = Xb.shape
    Xb_miss = []  # collect the noisy data
    for b in range(B):
        X = Xb[b].copy()  # M x D
        # Unroll X to 1D array: M*D
        X = X.reshape(-1)
        # Get the indices to mask/add noise
        mask_indices = np.random.choice(
            np.arange(X.size), replace=False, size=int(X.size * dropout)
        )
        # Introduce missing values as NaNs
        X[mask_indices] = np.nan
        # Reshape into the original dimensions
        X = X.reshape(M, D)
        Xb_miss.append(X)
    return np.array(Xb_miss)


######################################################################


# Functions to process data #####################
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramer's V statistic for categorical-categorical association.

    Cramer's V is used to assess the association between two categorical variables.
    The output is in the range of [0,1], where 0 means no association and 1 means full
    association. Chi-square = 0 implies Cramér’s V = 0.

    Source: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Cramer's V is used with accounting for Bias correction.

    Note: chi-square = 0 implies that Cramér’s V = 0

    Args:
        x (pd.Series): A categorical variable.
        y (pd.Series): Another categorical variable.

    Returns:
        float: Cramer's V statistic, indicating the strength of association.
    """
    confusion_matrix = pd.crosstab(x, y)  # Create contingency table
    chi2 = chi2_contingency(confusion_matrix)[0]  # Compute chi-square statistic
    n = confusion_matrix.sum().sum()  # Total number of observations
    phi2 = chi2 / n  # Phi-square statistic
    r, k = confusion_matrix.shape  # Number of rows and columns
    # Bias correction for phi2
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)  # Bias correction for rows
    kcorr = k - ((k - 1) ** 2) / (n - 1)  # Bias correction for columns
    num = phi2corr  # Numerator for Cramer's V
    denom = min(kcorr - 1, rcorr - 1)  # Denominator for Cramer's V
    if denom == 0:
        return 0  # No association
    return np.sqrt(num / denom)  # Return Cramer's V


def correlation_ratio(categories: pd.Series, measurements: np.ndarray) -> float:
    """Find the correlation ratio between categorical and numerical features.

    Correlation ratio is used to measure the relationship between a categorical
    variable and a numerical variable. The result ranges from 0 to 1.

    Source: https://en.wikipedia.org/wiki/Correlation_ratio

    Args:
        categories (pd.Series): Categorical variable.
        measurements (np.ndarray): Numerical measurements corresponding to categories.

    Returns:
        float: The correlation ratio (eta).
    """
    fcat, _ = pd.factorize(categories)  # Factorize categories to numeric values
    cat_num = np.max(fcat) + 1  # Number of unique categories
    y_avg_array = np.zeros(cat_num)  # Store category-wise averages
    n_array = np.zeros(cat_num)  # Store category-wise counts
    for i in range(cat_num):
        cat_measures = measurements[
            np.argwhere(fcat == i).flatten()
        ]  # Get measurements
        n_array[i] = len(cat_measures)  # Count of samples in the category
        y_avg_array[i] = np.average(cat_measures)  # Average for the category
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(
        n_array
    )  # Total average
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(
        np.power(np.subtract(measurements, y_total_avg), 2)
    )  # Total variance
    if numerator == 0:
        eta = 0.0  # No correlation if numerator is zero
    else:
        eta = np.sqrt(numerator / denominator)  # Calculate correlation ratio
    return eta


def pairwise_cov_matrix(df: pd.DataFrame, dtype: dict[str, str]) -> pd.DataFrame:
    """Calculate the covariance matrix using pairwise calculations.
    Accounts for categorical, numerical & Real features.
    `Cat-Cat' association is calculated using cramers V statistic.
    `Cat-Num' value is obtained using the correlation ratio.
    `Num-Num' correlation is calculated using the Pearson coefficient.

    Args:
        df (pd.DataFrame): The input data with shape M(samples) x D(features).
        dtype (Dict[str, str]): Dictionary specifying the type of each column
            (e.g., 'r' for real/numerical, 'c' for categorical).

    Returns:
        pd.DataFrame: The covariance matrix of shape D x D.
    """
    features = df.columns  # Extract feature names
    D = len(features)  # Number of features
    cov = np.zeros((D, D))  # Initialize covariance matrix with zeros
    for i, fi in enumerate(features):
        print(f"row feature {i, fi}")
        for j, fj in enumerate(features):
            if j >= i:  # Calculate only for upper triangle due to symmetry
                if dtype[fi] == "c" and dtype[fj] == "c":
                    cov[i, j] = cramers_v(df[fi], df[fj])  # Cramer's V for Cat-Cat
                elif dtype[fi] == "c" and dtype[fj] == "r":
                    cov[i, j] = correlation_ratio(df[fi], df[fj])  # Cat-Num correlation
                elif dtype[fi] == "r" and dtype[fj] == "c":
                    cov[i, j] = correlation_ratio(df[fj], df[fi])  # Cat-Num correlation
                elif dtype[fi] == "r" and dtype[fj] == "r":
                    cov[i, j] = pearsonr(df[fi], df[fj])[0]  # Pearson for Num-Num
                cov[j, i] = cov[i, j]  # Covariance is symmetric
    cov = pd.DataFrame(cov, index=features, columns=features)  # Convert to Dataframe
    return cov


def convert_to_torch(
    data: np.ndarray, req_grad: bool = False, use_cuda: bool = False
) -> torch.Tensor:
    """Convert data from numpy to torch tensor, enabling gradient calculation if
    `req_grad` is True.

    Args:
        data (np.ndarray): The input data to convert.
        req_grad (bool, optional): Whether to enable gradient calculation (default: False).
        use_cuda (bool, optional): Whether to use CUDA (GPU) for tensor computation
            (default: False).

    Returns:
        torch.Tensor: The converted torch tensor.
    """
    if not torch.is_tensor(data):  # Check if data is already a torch tensor
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  # Set dtype
        data = torch.from_numpy(data.astype(float, copy=False)).type(dtype)  # Convert
    data.requires_grad = req_grad  # Enable/disable gradient calculation
    return data


def eigen_val_condition_num(A: np.ndarray) -> tuple[list[float], float]:
    """Calculates the eigenvalues and the condition number of the input matrix A.

    Condition number is defined as: max(|eig|) / min(|eig|).

    Args:
        A (np.ndarray): The input matrix for which eigenvalues and condition number
            are to be calculated.

    Returns:
        Tuple[List[float], float]: A tuple containing the list of eigenvalues and the
            condition number of matrix A.
    """
    eig = [v.real for v in np.linalg.eigvals(A)]  # Compute real eigenvalues
    condition_number = max(np.abs(eig)) / min(np.abs(eig))  # Compute condition number
    return eig, condition_number


def get_covariance(Xb: np.ndarray, offset: float = 0.1) -> np.ndarray:
    """Calculate the batch covariance matrix for a batch of sample matrices.

    Args:
        Xb (3D np.ndarray): The input batch of sample matrices (B x M x D), where B
            is the batch size, M is the number of samples, and D is the number of features.
        offset (float, optional): The eigenvalue offset to adjust for a poor condition
            number (default: 0.1).

    Returns:
        np.ndarray: The batch covariance matrices (B x D x D).
    """
    Sb = []  # List to store covariance matrices for each batch
    for X in Xb:
        S = covariance.empirical_covariance(
            X, assume_centered=False
        )  # Covariance matrix
        eig, con = eigen_val_condition_num(
            S
        )  # Compute eigenvalues and condition number
        if min(eig) <= 1e-6:  # Check if the condition number is poor
            print(f"Adjust the eval: min {min(eig)}, con {con}")
            S += np.eye(S.shape[-1]) * (offset - min(eig))  # Adjust eigenvalues
            eig, con = eigen_val_condition_num(
                S
            )  # Recompute eigenvalues and condition number
            print(f"new eval: min {min(eig)}, con {con}")
        Sb.append(S)  # Append adjusted covariance matrix to the list
    return np.array(Sb)  # Return the batch of covariance matrices


# Functions to check the input ########
# Processing the input data to be compatiable for the sparse graph recovery models
def process_table(
    table: pd.DataFrame,
    NORM: str = "no",
    MIN_VARIANCE: float = 0.0,
    msg: str = "",
    COND_NUM: float = np.inf,
    eigval_th: float = 1e-3,
    VERBOSE: bool = True,
) -> pd.DataFrame:
    """Processing the input data to be compatiable for the
    sparse graph recovery models. Checks for the following
    issues in the input tabular data (real values only).
    Note: The order is important. Repeat the function
    twice: process_table(process_table(table)) to ensure
    the below conditions are satisfied.
    1. Remove all the rows with zero entries
    2. Fill Nans with column mean
    3. Remove columns containing only a single entry
    4. Remove columns with duplicate values
    5. Remove columns with low variance after centering
    The above steps are taken in order to ensure that the
    input matrix is well-conditioned.

    Args:
        table (pd.DataFrame): The input table with headers.
        NORM (str, optional): Normalization method ('min_max', 'mean', 'no'). Default is 'no'.
        MIN_VARIANCE (float, optional): Minimum variance threshold to drop columns
            with low variance. Default is 0.0.
        msg (str, optional): A custom message to print during processing. Default is ''.
        COND_NUM (float, optional): The maximum condition number allowed. Default is np.inf.
        eigval_th (float, optional): Minimum eigenvalue threshold for removing highly correlated
            columns. Default is 1e-3.
        VERBOSE (bool, optional): Whether to print progress messages. Default is True.

    Returns:
        pd.DataFrame: The processed table with headers, ready for sparse graph recovery models.
    """
    start = time()
    if VERBOSE:
        print(f"{msg}: Processing the input table for basic compatibility check")
        print(
            f"{msg}: The input table has sample {table.shape[0]} and features {table.shape[1]}"
        )

    total_samples = table.shape[0]

    # typecast the table to floats
    # table = table._convert(numeric=True)
    table = table.astype(float)

    # 1. Removing all the rows with zero entries as the samples are missing
    table = table.loc[~(table == 0).all(axis=1)]
    if VERBOSE:
        print(f"{msg}: Total zero samples dropped {total_samples - table.shape[0]}")

    # 2. Fill nan's with mean of columns
    table = table.fillna(table.mean())

    # 3. Remove columns containing only a single value
    single_value_columns = []
    for col in table.columns:
        if len(table[col].unique()) == 1:
            single_value_columns.append(col)
    table.drop(single_value_columns, inplace=True, axis=1)
    if VERBOSE:
        print(
            f"{msg}: Single value columns dropped:\
            total {len(single_value_columns)}, columns {single_value_columns}"
        )

    # Normalization of the input table
    table = normalize_table(table, NORM)

    # Analysing the input table's covariance matrix condition number
    analyse_condition_number(table, "Input", VERBOSE)

    # 4. Remove columns with duplicate values
    all_columns = table.columns
    table = table.T.drop_duplicates().T
    duplicate_columns = list(set(all_columns) - set(table.columns))
    if VERBOSE:
        print(
            f"{msg}: Duplicates dropped:\
                total {len(duplicate_columns)}, columns {duplicate_columns}"
        )

    # 5. Columns having similar variance have a slight chance of being almost duplicates
    # which can affect the condition number of the covariance matrix.
    # Also columns with low variance are less informative
    table_var = table.var().sort_values(ascending=True)
    # print(f'{msg}: Variance of the columns {table_var.to_string()}')
    # Dropping the columns with variance < MIN_VARIANCE
    low_variance_columns = list(table_var[table_var < MIN_VARIANCE].index)
    table.drop(low_variance_columns, inplace=True, axis=1)
    if VERBOSE:
        print(
            f"{msg}: Low Variance columns dropped: min variance {MIN_VARIANCE},\
        total {len(low_variance_columns)}, columns {low_variance_columns}"
        )

    # Analysing the processed table's covariance matrix condition number
    cov_table, eig, con = analyse_condition_number(table, "Processed", VERBOSE)

    itr = 1
    while con > COND_NUM:  # ill-conditioned matrix
        if VERBOSE:
            print(
                f"{msg}: {itr} Condition number is high {con}. \
            Dropping the highly correlated features in the cov-table"
            )
        # Find the number of eig vals < eigval_th for the cov_table matrix.
        # Rough indicator of the lower bound num of features that are highly correlated.
        eig = np.array(sorted(eig))
        lb_ill_cond_features = len(eig[eig < eigval_th])
        if VERBOSE:
            print(
                f"Current lower bound on ill-conditioned features {lb_ill_cond_features}"
            )
        if lb_ill_cond_features == 0:
            if VERBOSE:
                print(f"All the eig vals are > {eigval_th} and current cond num {con}")
            if con > COND_NUM:
                lb_ill_cond_features = 1
            else:
                break
        highly_correlated_features = get_highly_correlated_features(cov_table)
        # Extracting the minimum num of features making the cov_table ill-conditioned
        highly_correlated_features = highly_correlated_features[
            : min(lb_ill_cond_features, len(highly_correlated_features))
        ]
        # The corresponding column names
        highly_correlated_columns = table.columns[highly_correlated_features]
        if VERBOSE:
            print(
                f"{msg} {itr}: Highly Correlated features dropped {highly_correlated_columns}, \
        {len(highly_correlated_columns)}"
            )
        # Dropping the columns
        table.drop(highly_correlated_columns, inplace=True, axis=1)
        # Analysing the processed table's covariance matrix condition number
        cov_table, eig, con = analyse_condition_number(
            table,
            f"{msg} {itr}: Corr features dropped",
            VERBOSE,
        )
        # Increasing the iteration number
        itr += 1
    if VERBOSE:
        print(
            f"{msg}: The processed table has sample\
                {table.shape[0]} and features {table.shape[1]}"
        )
        print(
            f"{msg}: Total time to process the table {np.round(time()-start, 3)} secs"
        )
    return table


def get_highly_correlated_features(input_cov: np.ndarray) -> np.ndarray:
    """Taking the covariance of the input covariance matrix to find the highly
    correlated features that makes the input cov matrix ill-conditioned.
    Args:
        input_cov (np.ndarray): 2D covariance matrix of shape (D, D).

    Returns:
        np.ndarray: List of indices of highly correlated features to drop.
    """
    cov2 = covariance.empirical_covariance(input_cov)
    # mask the diagonal
    np.fill_diagonal(cov2, 0)
    # Get the threshold for top 10%
    cov_upper = upper_tri_indexing(np.abs(cov2))
    sorted_cov_upper = [
        i for i in sorted(enumerate(cov_upper), key=lambda x: x[1], reverse=True)
    ]
    th = sorted_cov_upper[int(0.1 * len(sorted_cov_upper))][1]
    # Getting the feature correlation dictionary
    high_indices = np.transpose(np.nonzero(np.abs(cov2) >= th))
    high_indices_dict = {}
    for i in high_indices:  # the upper triangular part
        if i[0] in high_indices_dict:
            high_indices_dict[i[0]].append(i[1])
        else:
            high_indices_dict[i[0]] = [i[1]]
    # sort the features based on the number of other correlated features.
    top_correlated_features = [[f, len(v)] for (f, v) in high_indices_dict.items()]
    top_correlated_features.sort(key=lambda x: x[1], reverse=True)
    top_correlated_features = np.array(top_correlated_features)
    features_to_drop = top_correlated_features[:, 0]
    return features_to_drop


def upper_tri_indexing(A: np.ndarray) -> np.ndarray:
    """Extract the upper triangular elements of a square matrix (excluding diagonal).

    Args:
        A (np.ndarray): A square matrix of shape (m, m).

    Returns:
        np.ndarray: A 1D array containing the upper triangular elements of the matrix.
    """
    m = A.shape[0]  # Get the number of rows (or columns) of the matrix
    r, c = np.triu_indices(
        m, 1
    )  # Get indices for the upper triangle (excluding diagonal)
    return A[r, c]  # Return the elements at those indices


def analyse_condition_number(
    table: pd.DataFrame, MESSAGE: str = "", VERBOSE: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Analyze the covariance matrix of a table to compute the condition number and eigenvalues.

    Args:
        table (pd.DataFrame): Input data table (samples x features).
        MESSAGE (str, optional): A message to display when VERBOSE is True.
        VERBOSE (bool, optional): Whether to print condition number and eigenvalue info.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - Covariance matrix (np.ndarray).
            - Eigenvalues of the covariance matrix (np.ndarray).
            - Condition number (float).
    """
    S = covariance.empirical_covariance(
        table, assume_centered=False
    )  # Covariance matrix
    eig, con = eigen_val_condition_num(S)  # Eigenvalues and condition number
    if VERBOSE:
        print(
            f"{MESSAGE} covariance matrix: The condition number {con} and "
            f"min eig {min(eig)} max eig {max(eig)}"
        )
    return S, eig, con  # Return covariance matrix, eigenvalues, and condition number


def normalize_table(df: pd.DataFrame, typeN: str) -> pd.DataFrame:
    """Normalize the input table based on the specified normalization type.

    Args:
        df (pd.DataFrame): Input data table (samples x features).
        typeN (str): Type of normalization ('min_max', 'mean', or 'no').

    Returns:
        pd.DataFrame: The normalized table.
    """
    if typeN == "min_max":
        return (df - df.min()) / (df.max() - df.min())  # Min-max normalization
    elif typeN == "mean":
        return (df - df.mean()) / df.std()  # Z-score (mean) normalization
    else:
        print(f"No Norm applied: Type entered {typeN}")  # No normalization
        return df  # Return the original table if no normalization