###############################################################################
# File:         model.py
# Description:  Diffrent example how to build a model in pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      Marche 12, 2023
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
from pytagi import NetProp


class RegressionMLP(NetProp):
    """Multi-layer perceptron for regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1]
        self.nodes = [1, 50, 1]
        self.activations = [0, 4, 0]
        self.batch_size = 4
        self.sigma_v = 0.06
        self.sigma_v_min: float = 0.06
        self.device = "cpu"


class HeterosMLP(NetProp):
    """Multi-layer preceptron for regression task where the
    output's noise varies overtime"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1]
        self.nodes: list = [1, 128, 2]  # output layer = [mean, std]
        self.activations: list = [0, 4, 0]
        self.batch_size: int = 1
        self.sigma_v: float = 0
        self.sigma_v_min: float = 0
        self.noise_type: str = "heteros"
        self.noise_gain: float = 1.0
        self.init_method: str = "He"
        self.device: str = "cpu"