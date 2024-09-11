### Developed By: Shaik Shoaib Nawaz
### Register No: 212222240094

# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
### Date: 

### AIM:
To Compute the AutoCorrelation Function (ACF) of the power Consumption dataset and 
to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(0)
data = pd.read_csv('/content/KwhConsumptionBlower78_1.csv')
data['TxnDate'] = pd.to_datetime(data['TxnDate'])
data = data.sort_values(by='TxnDate') 
data.set_index('TxnDate', inplace=True)
data.dropna(inplace=True)
plt.figure(figsize=(12, 6))
plt.plot( data['Consumption'], label='Data')
plt.xlabel('TxnDate')
plt.ylabel('Consumption')
plt.legend()
plt.title('Consumption Data')
plt.show()
data.dropna(inplace=True)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
y_train = train_data['Consumption']
y_test = test_data['Consumption']
from statsmodels.graphics.tsaplots import plot_acf
series = data['Consumption']
plot_acf(series)
plt.show()
data['Consumption'].corr(data['Consumption'].shift(1))
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
lag_order = 35 
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()
import statistics

y_pred = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
variance = np.var(y_test)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Variance_testing: {variance:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(test_data["Consumption"] ,y_test, label='Actual Consumption')
plt.plot( test_data["Consumption"],y_pred, label='Predicted Consumption', linestyle='--')
plt.xlabel('TxnDate')
plt.ylabel('Consumption')
plt.legend()
plt.title('Consumption Prediction with Autoregressive Model')
plt.show()
```

### OUTPUT:
#### VISUAL REPRESENTATION OF DATASET:
![dataset](image1.png)

#### AUTO CORRELATION:
![auto_correlation](image2.png)
![VALUE_CORRELATION](image3.png)

#### VALUES OF MAE,RMSE,VARIANCE:
![MEAN](image4.png)

#### AUTOREGRESSIVE MODEL FOR CONSUMPTION PREDICTION
![GRAPH](image5.png)

### RESULT: 
Thus, The python code for implementing auto correlation for power consumption is successfully executed.
