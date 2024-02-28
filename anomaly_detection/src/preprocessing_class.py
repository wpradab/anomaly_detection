import numpy as np
import pandas as pd
from typing import Literal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, FactorAnalysis


class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def drop_null_columns(self, id_column: Literal["DBInstanceIdentifier", "DBClusterIdentifier"]) -> pd.DataFrame:
        df_clean = self.df.dropna(axis=1, how='all')
        cols_to_drop = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
        cols_to_drop = [col for col in cols_to_drop if (col != id_column)]
        df_clean = df_clean.drop(columns=cols_to_drop, axis=1)
        return df_clean

    def drop_redundant_cols(self, threshold: float = 0.95) -> pd.DataFrame:
        numeric_data = self.df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_data.corr()
        redundant_columns = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    redundant_columns.add(correlation_matrix.columns[j])
        data_without_redundancy = self.df.drop(columns=redundant_columns)
        return data_without_redundancy

    def filter_nan_cols(self, threshold: float = 0.5) -> pd.DataFrame:
        max_nan = len(self.df) * threshold
        nan_per_column = self.df.isna().sum()
        columns_to_drop = nan_per_column[nan_per_column > max_nan].index
        filtered_data = self.df.drop(columns=columns_to_drop, axis=1)
        filtered_data = filtered_data.fillna(filtered_data.mean())
        return filtered_data

    def standardize_data(self, id_column: Literal["DBInstanceIdentifier", "DBClusterIdentifier"], threshold: float = 0.5) -> pd.DataFrame:
        categorical_column = self.df[id_column]
        numeric_data = self.df.select_dtypes(include=['float64', 'int64'])
        numeric_data = self.filter_nan_cols(threshold)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        scaled_data_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        scaled_data_df[id_column] = categorical_column.reset_index(drop=True)
        return scaled_data_df

    def dataframe_pca_transform(self, new_dimension: int) -> np.ndarray:
        num_data = self.df.select_dtypes(include=['float64', 'int64'])
        pca = PCA(n_components=new_dimension)
        principal_components = pca.fit_transform(num_data)
        return principal_components

    def ica_dimensionality_reduction_plot(self, new_dimension: int) -> np.ndarray:
        df = self.df.select_dtypes(include=['float64', 'int64'])
        df = df.values
        ica = FastICA(n_components=new_dimension, random_state=42)
        reduced_features = ica.fit_transform(df)
        return reduced_features

    def factor_analysis_and_plot(self, new_dimension: int) -> np.ndarray:
        df = self.df.select_dtypes(include=['float64', 'int64'])
        if isinstance(df, pd.DataFrame):
            df = df.values
        fa = FactorAnalysis(n_components=new_dimension)
        factors = fa.fit_transform(df)
        return factors
