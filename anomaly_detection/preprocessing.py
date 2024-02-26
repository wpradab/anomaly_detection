import pandas as pd
from typing import Literal
from sklearn.preprocessing import StandardScaler


def drop_null_columns(
    df: pd.DataFrame,
    id_column: str
) -> pd.DataFrame:
    """
    Drop null columns and those with the same value in all records, except the identifier column.

    Args:
    - df (pd.DataFrame): Pandas DataFrame.
    - id_column (str): Name of the identifier column that should not be dropped.

    Returns:
    pd.DataFrame: Clean DataFrame.
    """
    df_clean = df.dropna(
        axis=1,
        how='all'
    )
    # Drop columns with the same value for all records
    cols_to_drop = [col for col in df_clean.columns if df_clean[col].nunique() == 1]

    # Do not include 'DBInstanceIdentifier' in columns to drop
    cols_to_drop = [col for col in cols_to_drop if (col != id_column)]

    df_clean = df_clean.drop(columns=cols_to_drop, axis=1)
    return df_clean


def drop_redundant_cols(
    df: pd.DataFrame,
    threshold: float = 0.95
):
    """
    Drop redundant columns from a DataFrame based on a correlation threshold.

    Args:
    - df_instances (pd.DataFrame): DataFrame with the data.
    - threshold (float): Correlation threshold. Columns with a correlation greater than this value are considered redundant.

    Returns:
    - pd.DataFrame: DataFrame without redundant columns.
    """
    # Filter only numeric columns
    numeric_data = df.select_dtypes(include=['float64', 'int64'])

    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()

    # Find columns with high correlation
    redundant_columns = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                redundant_columns.add(correlation_matrix.columns[j])

    # Drop redundant columns
    data_without_redundancy = df.drop(columns=redundant_columns)

    return data_without_redundancy


def filter_nan_cols(
    df: pd.DataFrame,
    threshold: float = 0.5
):
    """
    Filter columns that have a number of NaN values exceeding the given threshold.

    Args:
    - data (pd.DataFrame): Pandas DataFrame with the data.
    - threshold (float): Allowed NaN threshold as a fraction.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    # Calculate maximum number of allowed NaN
    max_nan = len(df) * threshold

    # Get NaN count per column
    nan_per_column = df.isna().sum()

    # Filter columns exceeding the threshold
    columns_to_drop = nan_per_column[nan_per_column > max_nan].index

    # Drop columns
    filtered_data = df.drop(columns=columns_to_drop, axis=1)

    # # Fill missing values if less than 50% of the data is missing
    # if nan_per_column.max() < max_nan:
    filtered_data = filtered_data.fillna(filtered_data.mean())

    return filtered_data


def standardize_data(
        df,
        id_column: Literal["DBInstanceIdentifier", "DBClusterIdentifier"],
        threshold=0.5
):
    """
    Standardize numeric data of a DataFrame and preserve a specified categorical column.

    Args:
    - data (pd.DataFrame): Pandas DataFrame with the data.
    - id_column (Literal["DBInstanceIdentifier", "DBClusterIdentifier"]): Name of the categorical column to preserve.
    - threshold (float): Allowed NaN threshold as a fraction.

    Returns:
    pd.DataFrame: DataFrame with standardized data.
    """
    categorical_column = df[id_column]
    # Select only numeric columns
    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    numeric_data = filter_nan_cols(numeric_data, threshold)

    # Scale data to have zero mean and unit variance
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Rebuild a DataFrame with standardized data
    scaled_data_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
    # Add the categorical column back to the DataFrame
    scaled_data_df['DBInstanceIdentifier'] = categorical_column.reset_index(drop=True)

    return scaled_data_df


