from anomaly_detection.utils.S3 import S3Manager

s3_manager = S3Manager('credenciales.json')

df_instances = s3_manager.load_data_from_s3('instance/')