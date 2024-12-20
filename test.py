import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

# Create a DataFrame from the provided dataset
data_1 = pd.read_csv('Produksi Buah–Buahan dan Sayuran Tahunan Menurut Jenis Tanaman di Kabupaten Jember, 2019.csv')
data_2 = pd.read_csv('Produksi Buah–Buahan dan Sayuran Tahunan Menurut Jenis Tanaman di Kabupaten Jember, 2020.csv').drop(columns=['Jenis Tanaman'],axis=1)
data_3 = pd.read_csv('Produksi Buah–Buahan dan Sayuran Tahunan Menurut Jenis Tanaman di Kabupaten Jember, 2021.csv').drop(columns=['Jenis Tanaman'],axis=1)
data_4 = pd.read_csv('Produksi Buah–Buahan dan Sayuran Tahunan Menurut Jenis Tanaman di Kabupaten Jember, 2022.csv').drop(columns=['Jenis Tanaman'],axis=1)
data_5 = pd.read_csv('Produksi Buah–Buahan dan Sayuran Tahunan Menurut Jenis Tanaman di Kabupaten Jember, 2023.csv').drop(columns=['Jenis Tanaman'],axis=1)
data = pd.concat([data_1,data_2,data_3,data_4,data_5],axis=1)
data.columns = ['Jenis Tanaman','2019','2020','2021','2022','2023']
data.replace('...', np.nan, inplace=True)
data.dropna(inplace=True)
df = data

# Prepare a list to store predictions
predictions_list = []

# Loop through each fruit to perform regression analysis and generate predictions
for index, row in df.iterrows():
    plant = row['Jenis Tanaman']
    y = row[1:].values  # Production values
    years = np.array([2019, 2020, 2021, 2022, 2023])  # Year valuesF

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(years.reshape(-1, 1), y)

    # Make predictions for the next 5 years
    future_years = np.array([2024, 2025, 2026, 2027, 2028])
    future_predictions = model.predict(future_years.reshape(-1, 1))
    data = {"Jenis Tanaman": plant}
    
    # Append predictions to the list
    for year, prediction in zip(future_years, future_predictions):
        data[f"{year}"] = prediction
    predictions_list.append(data)
    
# Create a DataFrame from the predictions list
predictions_df = pd.DataFrame(predictions_list)
predictions_df.index = range(1,len(predictions_df) + 1)

# Pivot the predictions DataFrame to have years as columns
# predictions_pivot = predictions_df.pivot(index='Jenis Tanaman', values='Predicted Production').reset_index()

# Display the predictions table with years as columns
print(tabulate(predictions_df,headers= predictions_df.columns,tablefmt="double_grid"))