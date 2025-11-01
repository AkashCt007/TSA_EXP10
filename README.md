# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 1/11/2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load IMDb dataset
data = pd.read_csv("IMDB Top 250 Movies (1).csv")

# Prepare data
data = data[['year', 'rating']].copy()
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data.dropna(inplace=True)

# Group by year and get average rating
data = data.groupby('year')['rating'].mean().reset_index()
data.columns = ['Year', 'Rating']
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Plot IMDb ratings over years
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Rating'], color='blue')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.title('IMDb Ratings Time Series')
plt.grid()
plt.show()

# Stationarity check
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Rating'])

# ACF and PACF plots
plot_acf(data['Rating'])
plt.show()
plot_pacf(data['Rating'])
plt.show()

# Split into train/test
train_size = int(len(data) * 0.8)
train, test = data['Rating'][:train_size], data['Rating'][train_size:]

# SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 3))
sarima_result = sarima_model.fit()

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)

# Evaluate
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual IMDb Ratings', color='blue')
plt.plot(test.index, predictions, color='red', label='Predicted Ratings')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.title('SARIMA Model Predictions for IMDb Ratings')
plt.legend()
plt.grid()
plt.show()

```

### OUTPUT:
<img width="1189" height="685" alt="image" src="https://github.com/user-attachments/assets/5be9e175-846b-42c2-9129-7783eead56ca" />
<img width="813" height="500" alt="image" src="https://github.com/user-attachments/assets/e72db448-18ec-4781-aec2-a5d6810de36e" />
<img width="776" height="515" alt="image" src="https://github.com/user-attachments/assets/99d3894d-e1b7-48c6-9025-c0865dd18bf5" />
<img width="1084" height="670" alt="image" src="https://github.com/user-attachments/assets/78578fda-3471-428d-a5cc-577aeb9dac8b" />




### RESULT:
Thus the program run successfully based on the SARIMA model.
