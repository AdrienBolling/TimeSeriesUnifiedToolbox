"""Define dataframe-related utilities for the TSUT Framework."""

import pandas as pd


def filter_columns(
    data: pd.DataFrame, requested_columns: list[str] | None
) -> pd.DataFrame:
    """Filter the DataFrame to keep only the requested columns.

    If requested_columns is None, keep all columns.
    """
    if requested_columns is None:
        return data

    missing_columns = [col for col in requested_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"The following requested columns are not in the DataFrame: {missing_columns}"
        )

    return data[requested_columns]

def filter_dtypes(
    data: pd.DataFrame, requested_dtypes: list[str] | None
) -> pd.DataFrame:
    """Filter the DataFrame to keep only the columns with the requested dtypes.

    If requested_dtypes is None, keep all columns.
    """
    if requested_dtypes is None:
        return data

    selected_columns = data.select_dtypes(include=requested_dtypes).columns.tolist()
    if not selected_columns:
        raise ValueError(
            f"No columns in the DataFrame have the requested dtypes: {requested_dtypes}"
        )

    return data[selected_columns]
