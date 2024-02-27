import json
import boto3
import pickle
import pandas as pd
from typing import Union


class S3Manager:
    def __init__(
        self,
        credentials_file: str 
    ) -> None:
        """
        Initializes an S3Manager object.

        Args:
            credentials_file (str): Path to the credentials JSON file.

        Returns:
            None
        """
        with open(credentials_file, 'r') as file:
            credentials = json.load(file)

        self.aws_access_key_id = credentials['aws_access_key_id']
        self.aws_secret_access_key = credentials['aws_secret_access_key']
        self.region_name = credentials['region_name']
        self.bucket_name = credentials['bucket_name']

        self.s3_client = boto3.client('s3',
                                      aws_access_key_id=self.aws_access_key_id,
                                      aws_secret_access_key=self.aws_secret_access_key,
                                      region_name=self.region_name)

    def load_data_from_s3(
        self,
        object_key: str
    ) -> Union[pd.DataFrame, None]:
        """
        Loads data from an object in Amazon S3.

        Args:
            object_key (str): Key of the object in S3.

        Returns:
            Union[pd.DataFrame, None]: DataFrame if data loaded successfully, None if an error occurs.
        """
        try:
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=object_key)

            # Initialize a dictionary to store the dataframes
            dataframes = []

            # Check if objects were found
            for obj in objects.get('Contents', []):
                key = obj['Key']
                if key.endswith('.csv'):
                    obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                    df = pd.read_csv(obj['Body'])
                    dataframes.append(df)

            # Concatenate all DataFrames
            final_df = pd.concat(dataframes, ignore_index=True)
            return final_df
        except Exception as e:
            print("Error loading data from S3:", e)
            return None

    def save_data_to_s3(
        self,
        object_key: str,
        data
    ):
        """
        Saves data to an object in Amazon S3.

        Args:
            object_key (str): Key of the object in S3.
            data: Data to save to S3.

        Returns:
            None
        """
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=object_key, Body=data)
            print("Data saved to S3 successfully.")
        except Exception as e:
            print("Error saving data to S3:", e)

    def save_model_to_s3(
        self,
        object_key: str,
        model
    ):
        """
        Saves a model to an object in Amazon S3.

        Args:
            object_key (str): Key of the object in S3.
            model: Model to save to S3.

        Returns:
            None
        """
        try:
            # Serialize the model
            serialized_model = pickle.dumps(model)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=object_key, Body=serialized_model)
            print("Model saved to S3 successfully.")
        except Exception as e:
            print("Error saving model to S3:", e)
