import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    if df['Bankrupt?'].dtype == 'object':
        le = LabelEncoder()
        df['Bankrupt?'] = le.fit_transform(df['Bankrupt?'])
    X = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    df.to_csv(output_path, index=False)
    return X_scaled, y
