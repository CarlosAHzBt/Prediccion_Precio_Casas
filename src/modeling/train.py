import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    df = pd.read_csv('data/processed/california_housing_clean.csv')
    
    # Imprimir los nombres de las columnas para verificar
    print("Columnas del DataFrame:", df.columns)
    
    # Asegúrate de que la columna 'MedHouseVal' existe
    if 'MedHouseVal' not in df.columns:
        raise KeyError("La columna 'MedHouseVal' no se encuentra en el DataFrame.")
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Guardar el modelo entrenado
    joblib.dump(model, 'models/linear_regression_model.pkl')
    print("Modelo guardado en models/linear_regression_model.pkl")

if __name__ == "__main__":
    train_model()