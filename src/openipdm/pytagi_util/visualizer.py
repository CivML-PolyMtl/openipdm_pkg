###############################################################################
# File:         visualizer.py
# Description:  Visualization tool for images data
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      May 10, 2022
# Updated:      November 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
import os
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.typing as npt

plt.rcParams.update({
    "font.size": 18,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

class PredictionViz:
    """Visualization of prediction
    Attributes:
        task_name: Name of the task such as autoencoder
        data_name: Name of dataset such as Boston housing or toy example
        figsize: Size of figure
        fontsize: Font size for letter in the figure
        lw: linewidth
        ms: Marker size
        ndiv_x: Number of divisions for x-direction
        ndiv_y: Number of division for y-direciton
    """

    def __init__(self,
                 task_name: str,
                 data_name: str,
                 figsize: tuple = (12, 12),
                 fontsize: int = 28,
                 lw: int = 3,
                 ms: int = 10,
                 ndiv_x: int = 4,
                 ndiv_y: int = 4) -> None:
        self.task_name = task_name
        self.data_name = data_name
        self.figsize = figsize
        self.fontsize = fontsize
        self.lw = lw
        self.ms = ms
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def plot_predictions(self,
                         x_train: Union[np.ndarray, None],
                         y_train: Union[np.ndarray, None],
                         x_test: npt.NDArray,
                         y_test: npt.NDArray,
                         y_pred: npt.NDArray,
                         sy_pred: npt.NDArray,
                         std_factor: int,
                         sy_test: Union[np.ndarray, None] = None,
                         label: str = "diag",
                         title: Union[str, None] = None,
                         eq: Union[str, None] = None,
                         x_eq: Union[float, None] = None,
                         y_eq: Union[float, None] = None,
                         time_series: bool = False,
                         save_folder: Union[str, None] = None) -> None:
        """Compare prediciton distribution with theorical distribution

        x_train: Input train data
        y_train: Output train data
        x_test: Input test data
        y_test: Output test data
        y_pred: Prediciton of network
        sy_pred: Standard deviation of the prediction
        std_factor: Standard deviation factor
        sy_test: Output test's theorical standard deviation
        label: Name of file
        title: Figure title
        eq: Math equation for data
        x_eq: x-coordinate for eq
        y_eq: y-coordinate for eq

        """

        # Get max and min values
        if sy_test is not None:
            std_y = max(sy_test)
        else:
            std_y = 0

        if x_train is not None:
            max_y = np.maximum(max(y_test), max(y_train)) + std_y
            min_y = np.minimum(min(y_test), min(y_train)) - std_y
            max_x = np.maximum(max(x_test), max(x_train))
            min_x = np.minimum(min(x_test), min(x_train))
        else:
            max_y = max(y_test) + std_y
            min_y = min(y_test) - std_y
            max_x = max(x_test)
            min_x = min(x_test)

        # Plot figure
        plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set_title(title, fontsize=1.1 * self.fontsize, fontweight="bold")
        if eq is not None:
            ax.text(x_eq, y_eq, eq, color="k", fontsize=self.fontsize)
        idx = np.argsort(x_test)
        x_test = x_test[idx]
        y_test = y_test[idx]
        y_pred = y_pred[idx]
        sy_pred = sy_pred[idx]
        ax.plot(x_test, y_pred, "r", lw=self.lw, label=r"$\mathbb{E}[Y^{'}]$")
        ax.plot(x_test, y_test, "k", lw=self.lw, label=r"$y_{true}$", marker = '*', linestyle = "", ms=0.5 * self.ms, alpha = 0.5)

        ax.fill_between(x_test,
                        y_pred - std_factor * sy_pred,
                        y_pred + std_factor * sy_pred,
                        facecolor="red",
                        alpha=0.3,
                        label=r"$\mathbb{{E}}[Y^{{'}}]\pm{}\sigma$".format(std_factor))
        if sy_test is not None:
            ax.fill_between(x_test,
                            y_test - std_factor * sy_test,
                            y_test + std_factor * sy_test,
                            facecolor="blue",
                            alpha=0.3,
                            label=r"$y_{{test}}\pm{}\sigma$".format(std_factor))
        if x_train is not None:
            if time_series:
                marker = ""
                line_style = "-"
            else:
                marker = "o"
                line_style = ""
            ax.plot(x_train,
                    y_train,
                    "b",
                    marker=marker,
                    mfc="none",
                    lw=self.lw,
                    ms=0.2 * self.ms,
                    linestyle=line_style,
                    label=r"$y_{train}$",
                    alpha=0.5)

        ax.set_xlabel(r"$x$", fontsize=self.fontsize)
        ax.set_ylabel(r"$y$", fontsize=self.fontsize)
        if time_series:
            x_ticks = pd.date_range(min_x, max_x, periods=self.ndiv_x).values
        else:
            x_ticks = np.linspace(min_x, max_x, self.ndiv_x)
        y_ticks = np.linspace(min_y, max_y, self.ndiv_y)
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both",
                       which="both",
                       direction="inout",
                       labelsize=self.fontsize)
        ax.legend(
            loc="upper right",
            edgecolor="black",
            fontsize=1 * self.fontsize,
            ncol=1,
            framealpha=0.3,
            frameon=False,
        )
        ax.set_ylim([min_y, max_y])
        ax.set_xlim([min_x, max_x])

        # Save figure
        if save_folder is None:
            plt.show()
        else:
            saving_path = os.path.join(save_folder, f"pred_{label}_{self.data_name}.png")
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()