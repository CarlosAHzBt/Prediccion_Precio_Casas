import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("Datos originales:")
    print(df.head())

    # Ejemplo de preprocesamiento: eliminar valores nulos
    df = df.dropna()
    print("Datos después de eliminar valores nulos:")
    print(df.head())

    # Guardar datos procesados
    df.to_csv('data/processed/california_housing_clean.csv', index=False)
    print("Datos procesados guardados en data/processed/california_housing_clean.csv")

if __name__ == "__main__":
    preprocess_data('data/raw/california_housing.csv')
