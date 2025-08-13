from pprint import pprint

import numpy as np
from sklearn import metrics


def get_auc(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute the AUC and AUPR scores.

    Args:
        y (np.ndarray): Ground truth binary labels (1D array).
        scores (np.ndarray): Predicted scores or probabilities (1D array).

    Returns:
        Tuple[float, float]: AUC (Area Under the ROC Curve) and AUPR (Average
                            Precision) scores.
    """
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr


def report_metrics_all(
    trueG: np.ndarray, G: np.ndarray, beta: int = 1
) -> dict[str, float]:
    """Compute various metrics
    Args:
        trueG (np.ndarray): Ground truth precision matrix (2D array of floats).
        G (np.ndarray): Predicted precision matrix (2D array of floats).
        beta (int, optional): Beta for the F-beta score (default is 1).

    Returns:
        Dict: {fdr (float): (false positive) / prediction positive = FP/P
                tpr (float): (true positive) / condition positive = TP/T
                fpr (float): (false positive) / condition negative = FP/F
                shd (int): undirected extra + undirected missing = E+M
                nnz (int): number of non-zeros for trueG and predG
                ps (float): probability of success, sign match
                Fbeta (float): F-score with beta
                aupr (float): area under the precision-recall curve
                auc (float): area under the ROC curve}
    """
    trueG = trueG.real
    G = G.real
    # trueG and G are numpy arrays
    # convert all non-zeros in G to 1
    d = G.shape[-1]

    # changing to 1/0 for TP and FP calculations
    G_binary = np.where(G != 0, 1, 0)
    trueG_binary = np.where(trueG != 0, 1, 0)
    # extract the upper diagonal matrix
    indices_triu = np.triu_indices(d, 1)
    trueEdges = trueG_binary[indices_triu]  # np.triu(G_true_binary, 1)
    predEdges = G_binary[indices_triu]  # np.triu(G_binary, 1)
    # Getting AUROC value
    predEdges_auc = G[indices_triu]  # np.triu(G_true_binary, 1)
    auc, aupr = get_auc(trueEdges, np.absolute(predEdges_auc))
    # Now, we have the edge array for comparison
    # true pos = pred is 1 and true is 1
    TP = np.sum(trueEdges * predEdges)  # true_pos
    # False pos = pred is 1 and true is 0
    mismatches = np.logical_xor(trueEdges, predEdges)
    FP = np.sum(mismatches * predEdges)
    # Find all mismatches with Xor and then just select the ones with pred as 1
    # P = Number of pred edges : nnzPred
    P = np.sum(predEdges)
    nnzPred = P
    # T = Number of True edges :  nnzTrue
    T = np.sum(trueEdges)
    nnzTrue = T
    # F = Number of non-edges in true graph
    F = len(trueEdges) - T
    # SHD = total number of mismatches
    SHD = np.sum(mismatches)
    # FDR = False discovery rate
    FDR = FP / P
    # TPR = True positive rate
    TPR = TP / T
    # FPR = False positive rate
    FPR = FP / F
    # False negative = pred is 0 and true is 1
    FN = np.sum(mismatches * trueEdges)
    # F beta score
    num = (1 + beta**2) * TP
    den = (1 + beta**2) * TP + beta**2 * FN + FP
    Fbeta = num / den
    # precision
    precision = TP / (TP + FP)
    # recall
    recall = TP / (TP + FN)
    metrics = {
        "FDR": FDR,
        "TPR": TPR,
        "FPR": FPR,
        "SHD": SHD,
        "nnzTrue": T,
        "nnzPred": P,
        "precision": precision,
        "recall": recall,
        "Fbeta": Fbeta,
        "aupr": aupr,
        "auc": auc,
    }
    # convert to float + round to 3 decimals
    return {k: round(float(v), 3) for k, v in metrics.items()}


def summarize_compare_theta(
    compare_dict_list: list[dict[str, float]], method_name: str = "Method Name"
) -> dict[str, tuple[float, float]]:
    """Summarize and compare results across multiple runs.

    Args:
        compare_dict_list (List[Dict[str, float]]): List of dictionaries where each
                                                    dictionary contains metrics for
                                                    each run.
        method_name (str, optional): Name of the method being compared (default is
                                    'Method Name').

    Returns:
        Dict[str, Tuple[float, float]]: A dictionary where each key is a metric
                                        and the value is a tuple with the mean and
                                        standard deviation of that metric across
                                        all runs.
    """
    avg_results = {}  # Initialize an empty dictionary to store average results
    # Initialize the avg_results dictionary with empty lists for each metric
    for key in compare_dict_list[0].keys():
        avg_results[key] = []
    total_runs = len(compare_dict_list)
    # Iterate through all the runs in the comparison list
    for cd in compare_dict_list:
        for key in cd.keys():
            avg_results[key].append(cd[key])
    # Calculate rounded mean and standard deviation for each metric
    for key, values in avg_results.items():
        mean_val = round(float(np.mean(values)), 3)
        std_val = round(float(np.std(values)), 3)
        avg_results[key] = (mean_val, std_val)

    print(f"Avg results for {method_name} (mean, std)\n")
    pprint(avg_results)
    print(f"\nTotal runs {total_runs}\n\n")
    return avg_results


def report_metrics(trueG: np.ndarray, G: np.ndarray, beta: int = 1) -> dict[str, float]:
    """Compute various metrics
    Args:
        trueG (np.ndarray): Ground truth precision matrix (2D array of floats).
        G (np.ndarray): Predicted precision matrix (2D array of floats).
        beta (int, optional): Beta for the F-beta score (default is 1).

    Returns:
        Dict: {
                Fbeta (float): F-score with beta
                aupr (float): area under the precision-recall curve
                auc (float): area under the ROC curve
            }
    """
    trueG = trueG.real
    G = G.real
    # trueG and G are numpy arrays
    # convert all non-zeros in G to 1
    d = G.shape[-1]
    # changing to 1/0 for TP and FP calculations
    G_binary = np.where(G != 0, 1, 0)
    trueG_binary = np.where(trueG != 0, 1, 0)
    # extract the upper diagonal matrix
    indices_triu = np.triu_indices(d, 1)
    trueEdges = trueG_binary[indices_triu]  # np.triu(G_true_binary, 1)
    predEdges = G_binary[indices_triu]  # np.triu(G_binary, 1)
    # Getting AUROC value
    predEdges_auc = G[indices_triu]  # np.triu(G_true_binary, 1)
    auc, aupr = get_auc(trueEdges, np.absolute(predEdges_auc))
    # Now, we have the edge array for comparison
    # true pos = pred is 1 and true is 1
    TP = np.sum(trueEdges * predEdges)  # true_pos
    # False pos = pred is 1 and true is 0
    mismatches = np.logical_xor(trueEdges, predEdges)
    FP = np.sum(mismatches * predEdges)
    # F = Number of non-edges in true graph
    # False negative = pred is 0 and true is 1
    FN = np.sum(mismatches * trueEdges)
    # F beta score
    num = (1 + beta**2) * TP
    den = (1 + beta**2) * TP + beta**2 * FN + FP
    Fbeta = num / den
    return {"Fbeta": Fbeta, "aupr": aupr, "auc": auc}