import ray.data as d
import pyarrow as pa

from typing import final, override
from abc import ABC, abstractmethod

"""
This file contains schema definitions for the data fed to the Toolbox-dependent modules.

If necessary, the desired schema can be obtained through Ray Data transformations.

At the start of any training / tuning / serve pipeline however, the data must have been formatted according to the schema defined below.
"""


class SchemaEnforcer(ABC):
    """
    Base class for schema enforcement.
    It is primarly made to be used through Ray Data transformations.

    See examples at : https://docs.ray.io/en/latest/data/transforming-data.html#stateful-transforms

    Can also be used through the enforce method.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the schema enforcer.
        """
        pass

    @abstractmethod
    def __call__(self, dataset: d.Dataset) -> d.Dataset:
        """
        Enforce the schema on the given Ray Dataset through a Ray Data Transformation.

        Args:
            dataset (d.Dataset): The Ray Dataset to enforce the schema on.

        Returns:
            d.Dataset: A Ray Dataset with the enforced schema.

        """

        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def enforce(self, dataset: d.Dataset) -> d.Dataset:
        """
        Provides the same functionality as __call__ but through a direct method call.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def print_schema(self) -> pa.Schema:
        """
        Print the schema enforced by this enforcer.
        """

        raise NotImplementedError("Subclasses must implement this method.")


@final
class TSSchemaEnforcer(SchemaEnforcer):
    """
    Enforces a schema for time series data.

    The schema is as follows:
        - "timestamp": datetime64[ns]
        - "value": float64
        - "group": string (optional)
    """

    def __init__(self):
        """
        Initialize the time series schema enforcer.
        """
        super().__init__()
        self.schema = {
            "timestamp": "datetime64[ns]",
            "value": "float64",
            "group": "string",  # Optional
        }

    @override
    def __call__(self, dataset: d.Dataset) -> d.Dataset:
        """
        Enforce the time series schema on the given Ray Dataset through a Ray Data Transformation.

        Args:
            dataset (d.Dataset): The Ray Dataset to enforce the schema on.

        Returns:
            d.Dataset: A Ray Dataset with the enforced time series schema.

        """
        # FIX : don't use pyarrow schema but Ray Schemas instead
        dataset_schema = dataset.schema()
        if dataset_schema is None:
            raise ValueError("Provided Dataset schema is None. Cannot enforce schema.")
        elif not isinstance(dataset_schema, pa.Schema):
            table = next(dataset.iter_batches(batch_size=1, batch_format="pyarrow"))
            dataset_schema: pa.Schema = table.schema

        expected_names = [f.name for f in self.print_schema()]
        actual_names = dataset_schema.names

        missing_columns = set(expected_names) - set(actual_names)

        if len(missing_columns) > 0:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        # TODO : Add logging in case of extra columns

        # Type checking
        exp_by_name = {f.name: f.type for f in self.print_schema()}
        act_by_name = {f.name: f.type for f in dataset_schema}

        for name in expected_names:
            exp_field = exp_by_name[name]
            act_field = act_by_name[name]

            if not act_field.types_equal(exp_field):
                raise TypeError(
                    f"Column '{name}' has incorrect type. Expected {exp_field}, got {act_field}."
                )

        return dataset

    @override
    def enforce(self, dataset: d.Dataset) -> d.Dataset:
        """
        Enforce the time series schema on the given Ray Dataset through a direct method call.

        Args:
            dataset (d.Dataset): The Ray Dataset to enforce the schema on.

        Returns:
            d.Dataset: A Ray Dataset with the enforced time series schema.

        """

        return self.__call__(dataset)

    @override
    def print_schema(self) -> pa.Schema:
        """
        Print the time series schema
        """
        return pa.schema(
            [
                pa.field("timestamp", pa.timestamp("ns")),
                pa.field("value", pa.float64()),
            ],
            metadata={
                "timestamp": "Timestamp of the record, must be of regular interval",
                "value": "Value of the record, must be numeric",
            },
        )
