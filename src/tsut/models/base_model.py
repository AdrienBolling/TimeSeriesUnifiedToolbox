"""
This file contains the base class representing a model of the TimeSeriesUnifiedToolbox.
"""

import ray.data as d
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory

from abc import ABC, abstractmethod
from pathlib import Path

from tsut.core.annotations import endpoint
from tsut.core.common import EnforcedDict, ArrayLike


class BaseModel(Trainable, ABC):
    """
    Summary :
        A base class to reimplement for each prediction model of the TimeSeriesUnifiedToolbox.
        It provides a unified interface wherever the model comes from, and makes it necessary to implement any function necessary to handle Tuning and Distributed Training through the Ray framework.
    """

    # ---- Initiliaztion ----#
    def __init__(self) -> None:
        """
        Summary :
            Initialization function for the BaseModel class.
        Args:

        Returns:
            None
        """
        # INFO : All child subclasses must expicitly call all Parent classes constructors to ensure proper initialization. (Trainable has a particular behavior at init, thus BaseModel.__init__() must be called first in the MRO)
        Trainable.__init__(self)
        ABC.__init__(self)

    # ---- Base API for basic usage of the model ----#

    @endpoint
    def universal_setup(self, config: EnforcedDict) -> None:
        """
        Summary :
            A function to setup the model with the given configuration.
            Although this method is high-level and user-facing, it relies on lower-level methods that need to be implemented in child classes.
            It is not supposed to be overridden in child classes (but may be in special cases).

        Args:
            param config (EnforcedDict): Configuration dictionary for setting up the model.

        Returns:
            None
        """
        self.setup(config)

    @endpoint
    def universal_predict(self, input_data: d.Dataset) -> d.Dataset:
        """
        Summary :
            A function to perform prediction on input data using the model.
            Although this method is high-level and user-facing, it relies on lower-level methods that need to be implemented in child classes.
            It is not supposed to be overridden in child classes (but may be in special cases).

        Args:
            param input_data (d.Dataset): Input data for prediction.

        Returns:
            type d.Dataset: Dataset containing the predictions.
        """
        self._prepare_model()
        prepared_data: ArrayLike = self._preprocess(input_data)
        output: ArrayLike = self._predict(prepared_data)
        formatted_output: d.Dataset = self._postprocess(output)
        return formatted_output

    # ---- Internal methods to be implemented in child classes ----#
    # INFO : These methods are to be reimplemented in the child classes to ensure proper functionality of the model.

    def _prepare_model(self) -> None:
        """
        Summary :
            Internal method to prepare the model for prediction.
            The default implementation is an identity function, but child classes may override it if necessary.

        Args:

        Returns:
            None
        """
        pass

    def _preprocess(self, input_data: d.Dataset) -> ArrayLike:
        """
        Summary :
            Internal method to preprocess the input data before prediction.
            The default implementation is an identity function, but child classes may override it if necessary.
        Args:
            param input_data (d.Dataset): Input data for preprocessing.
        Returns:
            type ArrayLike: Preprocessed data.
        """
        return input_data

    def _predict(self, prepared_data: ArrayLike) -> ArrayLike:
        """
        Summary :
            Internal method to perform prediction on the preprocessed data.
            The default implementation is an identity function, but child classes must override it to provide actual prediction logic.

        Args:
            param prepared_data (ArrayLike): Preprocessed data for prediction.

        Returns:
            type ArrayLike: Predictions made by the model.
        """
        return prepared_data

    def _postprocess(self, output: ArrayLike) -> d.Dataset:
        """
        Summary :
            Internal method to postprocess the model's output after prediction.
            The default implementation is an identity function, but child classes may override it if necessary.

        Args:
            param output (ArrayLike): Output data from the model.

        Returns:
            type d.Dataset: Postprocessed dataset.
        """
        return output

    # ---- Sub-API for Ray Tune integration ----#
    # INFO : These methods are to be reimplemented in the child classes to ensure proper integration with Ray Tune.

    @abstractmethod
    def setup(self, config: EnforcedDict) -> None:
        """
        Summary :
            Setup function to initialize the model with the given configuration.
            This method must be implemented in each child class.

        Args:
            param config (EnforcedDict): Configuration dictionary for setting up the model.

        Returns:
            None
        """

        raise NotImplementedError(
            "The setup method must be implemented in the child class."
        )

    @abstractmethod
    def step(self) -> EnforcedDict:
        """
        Summary :
            Core training step function to be implemented in each child class.
            The return values will be automatically passed to the loggers.
        Args:

        Returns:
            EnforcedDict: Dictionary containing training metrics to be logged.
        """
        raise NotImplementedError(
            "The step method must be implemented in the child class."
        )

    @abstractmethod
    def save_checkpoint(
        self, checkpoint_dir: str | Path, checkpoint_name: str, overwrite: bool = True
    ) -> str:
        """
        Summary :
            Checkpoint saving function to be implemented in each child class.

        Args:
            param checkpoint_dir (str | Path): Directory where to save the checkpoint.
            param checkpoint_name (str): Name of the checkpoint file.
            param overwrite (bool): Whether to overwrite an existing checkpoint with the same name. Defaults to True.

        Returns:
            type str: Path to the checkpoint file.
        """

        raise NotImplementedError(
            "The save_checkpoint method must be implemented in the child class."
        )

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Summary :
            Checkpoint loading function to be implemented in each child class.
        Args:
            param checkpoint_path (str | Path): Path to the checkpoint file.

        Returns:
            None
        """

        raise NotImplementedError(
            "The load_checkpoint method must be implemented in the child class."
        )

    def cleanup(self) -> None:
        """
        Summary :
            Cleanup function to reimplement if you are creating additional Ray Actors during the Tuning phase, to kill the subprocesses properly.
        Args:

        Returns:
            None
        """
        pass

    def default_resource_request(
        self,
        config: dict,
    ) -> PlacementGroupFactory:
        """
        Summary :
            Default resource request function for Ray Tune.
            This method can be overridden in child classes to specify custom resource requirements. By default, it requests 1 CPU.

        Args:
            param config (dict): Configuration dictionary.

        Returns:
            type PlacementGroupFactory: Resource request for Ray Tune.
        """
        return PlacementGroupFactory([{"CPU": 1}])

    # INFO : Additionnal methods for advanced Ray Tune usage can be added here, such as reset_config
