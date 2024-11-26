import torch
import random
import numpy as np

def set_seed(seed):
    """Set seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to calculate DCF
def compute_dcf(y_true, y_pred, C_miss=1, C_fa=1, P_target=0.5):
    """
    Computes the Detection Cost Function (DCF).
    Args:
        y_true (torch.Tensor): True labels (0s and 1s).
        y_pred (torch.Tensor): Predicted probabilities.
        C_miss (float): Cost of misses.
        C_fa (float): Cost of false alarms.
        P_target (float): Prior probability of the target class.
    Returns:
        float: DCF score.
    """
    y_pred = (y_pred >= 0.5).float()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate

    dcf = C_miss * P_target * fnr + C_fa * (1 - P_target) * fpr
    return dcf
