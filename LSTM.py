import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# #Load Data

company = 'BNB-USD'
# # start = dt.datetime(2012,1,1)
# # end = dt.datetime(2020,1,1)
# # data = web.DataReader(company, 'yahoo', start, end)

# end = '2021-01-01'
# start = '2016-01-01'
# data = yf.download('BTC-USD', start, end)

# Download data from 'Yahoo! Finance' using yfinance
df = yf.download('BNB-USD', start='2022-06-30', end='2023-01-01', interval='1h')

# group in 4Hours chunks.
data1 = df.groupby(pd.Grouper(freq='4H')).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                                  "Adj Close": "last"})

# Remove the NaN rows
data = data1.dropna(how='all')
# Label the dataframe columns
data.columns = ["Open", "High", "Low", "Close", "Adj Close"]

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['High'].values.reshape(-1,1))
prediction_days = 300
x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Build the Model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of the next high value
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

''''Test the Model Accuracy on Existing Data'''

# #Load Test Data
# test_start = '2021-01-01'
# test_end = datetime.today().strftime('%Y-%m-%d')
# test_data = yf.download('BTC-USD', test_start, test_end)

# Download data from 'Yahoo! Finance' using yfinance
df2 = yf.download('BNB-USD', start='2022-06-30', end='2023-08-20', interval='1h')

# group in 4Hours chunks.
data2 = df2.groupby(pd.Grouper(freq='4H')).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                                  "Adj Close": "last"})
# Remove the NaN rows
test_data = data2.dropna(how='all')
# Label the dataframe columns
test_data.columns = ["Open", "High", "Low", "Close", "Adj Close"]



actual_prices = test_data['High'].values
total_dataset = pd.concat((data['High'], test_data['High']), axis=0)
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make Predictions on Test Data
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#Plot the Test Predictions
plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

#Predict for end date
real_data = [model_inputs[len(model_inputs)+1 -prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction for next date- {prediction}")