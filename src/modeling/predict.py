import os
import pandas as pd
import joblib

def predict_new_data(new_data):
    model = joblib.load('models/linear_regression_model.pkl')
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    file_path = 'data/raw/california_housing.csv'
    
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no se encuentra.")
    
    new_data = pd.read_csv(file_path)
    
    # Eliminar la columna 'MedHouseVal' si está presente
    if 'MedHouseVal' in new_data.columns:
        new_data = new_data.drop('MedHouseVal', axis=1)
    
    preds = predict_new_data(new_data)
    print("Predicciones:")
    print(preds)