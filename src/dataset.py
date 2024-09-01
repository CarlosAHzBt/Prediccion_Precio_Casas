from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_and_save_california_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    df.to_csv('data/raw/california_housing.csv', index=False)
    print("Data saved to data/raw/california_housing.csv")

if __name__ == "__main__":
    load_and_save_california_data()