from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.utils import resample


def get_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    bootstrap: bool = False,
    **bootstrap_kwargs
) -> Union[float, Tuple[float, float, float]]:

    def bootstrap_auc(n_resamples: int = 1000) -> List[float]:

        main_sample = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})

        booted_auc = []
        booted_fpr, booted_tpr = [], []

        for _ in range(n_resamples):
            sample = resample(main_sample)
            fpr, tpr, _ = roc_curve(sample["y_true"], sample["y_proba"])
            booted_fpr.append(fpr)
            booted_tpr.append(tpr)
            roc_auc = auc(fpr, tpr)
            booted_auc.append(roc_auc)

        return booted_auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    if bootstrap:
        bootstraped_auc = bootstrap_auc(y_true, y_proba)

        alpha = bootstrap_kwargs["confidence_level"]
        left_bound = np.quantile(bootstraped_auc, alpha)
        right_bound = np.quantile(bootstraped_auc, 1 - alpha)

        return left_bound, roc_auc, right_bound

    return roc_auc


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pos_label: Union[int, float, bool, str] = 1,
    figsize: Tuple[int, int] = (6, 6)
) -> Figure:

    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--")

    ax.set_title("ROC curve")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")

    plt.close(fig)

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pos_label: Union[int, float, bool, str] = 1,
    figsize: Tuple[int, int] = (6, 6)
) -> Figure:

    precision, recall, _ = precision_recall_curve(
        y_true, y_proba, pos_label=pos_label)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, label=None)
    ax.plot([0, 1], [1, 0], alpha=0, label=None)

    baseline = np.sum(np.where(y_true == pos_label, 1, 0)) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], "k--", label="Baseline")

    ax.set_title("PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.legend()
    plt.tight_layout()

    plt.close(fig)

    return fig
