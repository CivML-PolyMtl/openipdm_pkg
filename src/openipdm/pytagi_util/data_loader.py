###############################################################################
# File:         dataloader.py
# Description:  Prepare data for neural networks
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      January 27, 2023
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pytagi import Normalizer


class DataloaderBase(ABC):
    """Dataloader template"""

    normalizer: Normalizer = Normalizer()

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def process_data(self) -> dict:
        raise NotImplementedError

    def create_data_loader(self, raw_input: np.ndarray, raw_output: np.ndarray) -> list:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        # Even indices
        even_indices = self.split_evenly(num_input_data, self.batch_size)

        if np.mod(num_input_data, self.batch_size) != 0:
            # Remider indices
            rem_indices = self.split_reminder(num_input_data, self.batch_size)
            even_indices.append(rem_indices)

        indices = np.stack(even_indices)
        input_data = raw_input[indices]
        output_data = raw_output[indices]
        dataset = []
        for x_batch, y_batch in zip(input_data, output_data):
            dataset.append((x_batch, y_batch))
        return dataset

    def gen_data_loader(
        self, raw_input: np.ndarray, raw_output: np.ndarray
    ) -> np.ndarray:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        num_batches = num_output_data // self.batch_size

        # Trim the inputs and outputs to fit neatly into batches
        inputs = raw_input[: num_batches * self.batch_size]
        outputs = raw_output[: num_batches * self.batch_size]

        # Reshape the inputs and outputs into batches
        inputs_batched = np.reshape(inputs, (self.batch_size, num_batches, -1))
        # -1 automatically determines output features
        outputs_batched = np.reshape(outputs, (self.batch_size, num_batches, -1))

        return (inputs_batched, outputs_batched)

    @staticmethod
    def split_data(data: np.ndarray, split_ratio: float = 0.8) -> dict:
        """Split data into training and test sets"""
        num_data = data.shape[0]
        end_train_idx = int(split_ratio * num_data)
        train_data = data[:end_train_idx]
        test_data = data[end_train_idx:]
        return train_data, test_data

    @staticmethod
    def load_data_from_csv(data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def split_evenly(num_data, chunk_size: int):
        """split data evenly"""
        indices = np.arange(int(num_data - np.mod(num_data, chunk_size)))

        return np.split(indices, int(np.floor(num_data / chunk_size)))

    @staticmethod
    def split_reminder(num_data: int, chunk_size: int):
        """Pad the reminder"""
        indices = np.arange(num_data)
        reminder_start = int(num_data - np.mod(num_data, chunk_size))
        num_samples = chunk_size - (num_data - reminder_start)
        random_idx = np.random.choice(indices, size=num_samples, replace=False)
        reminder_idx = indices[reminder_start:]

        return np.concatenate((random_idx, reminder_idx))


class RegressionDataLoader(DataloaderBase):
    """Load and format data that are feeded to the neural network.
    The user must provide the input and output data file in *csv"""

    def __init__(self, batch_size: int, num_inputs: int, num_outputs: int) -> None:
        super().__init__(batch_size)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def process_data(
        self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str
    ) -> dict:
        """Process data from the csv file"""

        # Load data
        x_train = self.load_data_from_csv(x_train_file)
        y_train = self.load_data_from_csv(y_train_file)
        x_test = self.load_data_from_csv(x_test_file)
        y_test = self.load_data_from_csv(y_test_file)

        # Normalizer
        x_mean, x_std = self.normalizer.compute_mean_std(
            np.concatenate((x_train, x_test))
        )
        y_mean, y_std = self.normalizer.compute_mean_std(
            np.concatenate((y_train, y_test))
        )

        x_train = self.normalizer.standardize(data=x_train, mu=x_mean, std=x_std)
        y_train = self.normalizer.standardize(data=y_train, mu=y_mean, std=y_std)
        x_test = self.normalizer.standardize(data=x_test, mu=x_mean, std=x_std)
        y_test = self.normalizer.standardize(data=y_test, mu=y_mean, std=y_std)

        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = self.create_data_loader(
            raw_input=x_test, raw_output=y_test
        )
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
        data_loader["y_norm_param_1"] = y_mean
        data_loader["y_norm_param_2"] = y_std

        return data_loader
