from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import binomtest
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


def plot_reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 8,
    plot_ci: bool = True,
    figsize: Tuple[int, int] = (12, 5)
) -> Optional[Figure]:

    y_true = pd.Series(y_true)

    while n_bins > 1:
        try:
            pred_probs_space = np.linspace(
                y_proba.min(), y_proba.max(), n_bins + 1)

            empirical_probs = []
            pred_probs_midpts = []
            conf_intervals = []

            for i in range(len(pred_probs_space) - 1):
                bin_filter = (y_proba > pred_probs_space[i]) & (
                    y_proba < pred_probs_space[i + 1])

                empirical_probs.append(y_true[bin_filter].mean())
                pred_probs_midpts.append(pred_probs_space[i:i+2].mean())

                if plot_ci:
                    nsuccess = int(y_true[bin_filter].sum())
                    nobs = len(y_true[bin_filter])
                    ci = binomtest(nsuccess, nobs).proportion_ci()
                    conf_intervals.append((ci[0], ci[1]))

            conf_intervals = list(zip(*conf_intervals))

            break
        except Exception:
            n_bins -= 1

    if n_bins == 1:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pred_probs_midpts, empirical_probs,
            lw=2, marker="o", label="model")
    if plot_ci:
        ax.plot(pred_probs_midpts,
                conf_intervals[0], lw=2, ls=":", c="lightgray", label=r"5%")
        ax.plot(pred_probs_midpts,
                conf_intervals[1], lw=2, ls=":", c="lightgray", label=r"95%")
    ax.plot([np.min(y_proba), np.max(y_proba)],
            [np.min(y_proba), np.max(y_proba)], ls="--", c="gray",
            label="ideal")

    ax.set_title("Reliability (calibration) curve")
    ax.set_xlabel("Predicted probs")
    ax.set_ylabel("Empirical probs")

    ax.legend()
    plt.tight_layout()

    plt.close(fig)

    return fig


def plot_proba_distribution(
    y_proba: np.ndarray,
    n_bins: int = 20,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:

    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    sns.distplot(y_proba, hist=True, bins=n_bins,
                 kde=True, color="darkblue", ax=ax[0])
    sns.ecdfplot(y_proba, stat="proportion", ax=ax[1])

    ax[0].set_title("Probability distribution")
    ax[1].set_title("Cumulative probability distribution")
    ax[0].set_xlabel("Predicted probs")
    ax[1].set_xlabel("Predicted probs")

    plt.close(fig)

    return fig
