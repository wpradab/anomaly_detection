import pickle
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler


def drop_null_columns(
    df: pd.DataFrame,
    id_column: Literal["DBInstanceIdentifier", "DBClusterIdentifier"],
) -> pd.DataFrame:
    """
    Drop null columns and those with the same value in all records,
    except the identifier column.

    Args:
    - df (pd.DataFrame): Pandas DataFrame.
    - id_column (str): Name of the identifier column that should not
    be dropped.

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
) -> pd.DataFrame:
    """
    Drop redundant columns from a DataFrame based on a correlation
    threshold.

    Args:
    - df_instances (pd.DataFrame): DataFrame with the data.
    - threshold (float): Correlation threshold. Columns with a
    correlation greater than this value are considered redundant.

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
) -> pd.DataFrame:
    """
    Filter columns that have a number of NaN values exceeding
    the given threshold.

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
    df: pd.DataFrame,
    id_column: Literal["DBInstanceIdentifier", "DBClusterIdentifier"],
    threshold: float = 0.5,
    save_path: str = None
) -> pd.DataFrame:
    """
    Standardize numeric data of a DataFrame and preserve a specified
    categorical column.

    Args:
    - df (pd.DataFrame): Pandas DataFrame with the data.
    - id_column (Literal["DBInstanceIdentifier",
    "DBClusterIdentifier"]): Name of the categorical column to
    preserve.
    - threshold (float): Allowed NaN threshold as a fraction.
    - save_path (str): Path to save the scaler model.

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
    scaled_data_df[id_column] = categorical_column.reset_index(drop=True)

    # Save the scaler model if a path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)

    return scaled_data_df


def dataframe_pca_transform(
        df: pd.DataFrame,
        new_dimension: int,
        save_path: str = None
) -> np.ndarray:
    """
    Transforms data from a DataFrame using PCA (Principal
    Component Analysis).

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing the original data.
    new_dimension : int
        The number of dimensions to reduce the data via PCA.
    save_path (str): Path to save the PCA model.

    Returns:
    --------
    Union[pd.DataFrame, np.ndarray]
        Depends on the choice of return:
        - If a DataFrame is returned, it contains the principal components
        computed via PCA.
        - If an ndarray is returned, it is the data transformed via PCA.
    """
    num_data = df.select_dtypes(include=['float64', 'int64'])
    # Perform PCA
    pca = PCA(n_components=new_dimension)
    principal_components = pca.fit_transform(num_data)

    # Save the PCA model if a path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(pca, f)

    return principal_components


def dataframe_ica_transform(
        df: pd.DataFrame,
        new_dimension: int,
        save_path: str = None
) -> np.ndarray:

    """
    Realiza la reducción de dimensionalidad utilizando Análisis de
    Componentes Independientes (ICA) y grafica el resultado en un
    gráfico de dispersión para dos dimensiones.

    Parámetros:
    X (array-like): Matriz de características de entrada.
    n_components (int): Número de componentes independientes a extraer.
    save_path (str): Path to save the ICA model.

    Devuelve:
    reduced_features (array-like): Matriz de características reducida.
    """
    df = df.select_dtypes(include=['float64', 'int64'])
    df = df.values
    ica = FastICA(n_components=new_dimension, random_state=42)
    reduced_features = ica.fit_transform(df)

    # Save the ICA model if a path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(ica, f)

    return reduced_features


def dataframe_factor_analysis_transform(
    df: pd.DataFrame,
    new_dimension: int,
    save_path: str = None
) -> np.ndarray:
    """
    Performs Factor Analysis and plots the factor loadings.

    Parameters:
    -----------
    df (array-like or DataFrame): Input feature matrix.
    new_dimension (int): Number of factors to extract.
    save_path (str): Path to save the Factor Analysis model.

    Returns:
    --------
    factors (DataFrame): Matrix of factor scores.
    """

    df = df.select_dtypes(include=['float64', 'int64'])

    if isinstance(df, pd.DataFrame):
        df = df.values

    # Realizar Análisis de Factores# Perform Factor Analysis
    fa = FactorAnalyzer(n_factors=new_dimension, rotation=None)
    fa.fit(df)

    # Save the Factor Analysis model if a path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(fa, f)

    # Get latent factors
    factors = fa.transform(df)

    return factors
