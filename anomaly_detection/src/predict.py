import pickle
from preprocessing import drop_null_columns, drop_redundant_cols

def load_transformer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def predict_anomalies(df):

    df = drop_null_columns(df, "DBInstanceIdentifier")
    df = drop_redundant_cols(df)
    scaler = load_transformer('D:/Cencosud/anomaly_detection/anomaly_detection/src/models/standarize_data.pkl')
    pca = load_transformer('D:/Cencosud/anomaly_detection/anomaly_detection/src/models/pca_transformer.pkl')

    transformed_data = scaler.transform(df)
    transformed_data = pca.transform(transformed_data)