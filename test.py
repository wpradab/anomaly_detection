import pandas as pd
# from anomaly_detection.utils.S3 import S3Manager
#
# s3_manager = S3Manager('credenciales.json')
#
# df_instances = s3_manager.load_data_from_s3('instance/')

from anomaly_detection.src.preprocessing import drop_null_columns, drop_redundant_cols, \
    standardize_data, dataframe_pca_transform, dataframe_ica_transform, \
    dataframe_factor_analysis_transform

df_instances = pd.read_csv("C:/Users/USER/Downloads/df_instance.csv")

df_instances2 = df_instances.copy()
for instance in df_instances2.DBInstanceIdentifier.unique():
    df_instance_temp = df_instances2[df_instances2.DBInstanceIdentifier.isin([instance])]
    df_instance_temp = drop_null_columns(df_instance_temp, "DBInstanceIdentifier")
    df_instance_temp = drop_redundant_cols(df_instance_temp)
    df_instance_temp = standardize_data(df_instance_temp, "DBInstanceIdentifier", threshold=0.5, save_path='D:/Cencosud/anomaly_detection/anomaly_detection/src/models/standarize_data.pkl')
    print(instance)
    pca = dataframe_pca_transform(df_instance_temp, 2, save_path='D:/Cencosud/anomaly_detection/anomaly_detection/src/models/pca_transformer.pkl')
    ica = dataframe_ica_transform(df_instance_temp, 2, save_path='D:/Cencosud/anomaly_detection/anomaly_detection/src/models/ica_transformer.pkl')
    factor_analysis = dataframe_factor_analysis_transform(df_instance_temp, 2, save_path='D:/Cencosud/anomaly_detection/anomaly_detection/src/models/fa_transformer.pkl')

