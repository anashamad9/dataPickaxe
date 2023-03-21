import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
plt.style.use('fivethirtyeight')

df = pd.read_csv("DataPickaxe5.csv",index_col='date',parse_dates=True)
data = df._convert(numeric=True)

data

temp = df['quantity']
temp.plot()

data.describe()

data.head(10)

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
data['weather']= label_encoder.fit_transform(data['weather'])
data['item name']= label_encoder.fit_transform(data['item name'])
data['weather'].unique()
#data['item name'].unique()

data

data = data.drop(['time'], axis=1)

data = data.astype(float)
print(data.dtypes)

"""
data['price'] = data['price'].astype(float)
data['year'] = data['year'].astype(float)
data['day'] = data['day'].astype(float)
data['quantity'] = data['quantity'].astype(float)
data['holiday'] = data['holiday'].astype(float)
data['weekend'] = data['weekend'].astype(float)
data['dbh'] = data['dbh'].astype(float)
data['dah'] = data['dah'].astype(float)
data['tempreture'] = data['tempreture'].astype(float)
data['offer'] = data['offer'].astype(float)
data['weather'] = data['weather'].astype(float)
data['item name'] = data['item name'].astype(float)
data['month'] = data['month'].astype(float)
data['time'] = data['time'].astype(float)
"""


def data_to_X_y(data, window_size=5):
  data_as_np = data.to_numpy()
  X = []
  y = []
  for i in range(len(data_as_np)-window_size):
    row = [[a] for a in data_as_np[i:i+window_size]]
    X.append(row)
    label = data_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


WINDOW_SIZE = 5
X1, y1 = data_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape

X_train1, y_train1 = X1[:80], y1[:80]
X_val1, y_val1 = X1[80:90], y1[80:90]
X_test1, y_test1 = X1[90:], y1[90:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()


cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])


from tensorflow.keras.models import load_model
model1 = load_model('model1/')


train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])


val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results


plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])


test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results


plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])

from sklearn.metrics import mean_squared_error as mse

def plot_predictions1(model, X, y, start=0, end=100):
  predictions = model.predict(X).flatten()
  df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
  plt.plot(df['Predictions'][start:end])
  plt.plot(df['Actuals'][start:end])
  return df, mse(y, predictions)


plot_predictions1(model1, X_test1, y_test1)


model2 = Sequential()
model2.add(InputLayer((5, 1)))
model2.add(Conv1D(64, kernel_size=2))
model2.add(Flatten())
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))

model2.summary()


cp2 = ModelCheckpoint('model2/', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model2.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp2])

model3 = Sequential()
model3.add(InputLayer((5, 1)))
model3.add(GRU(64))
model3.add(Dense(8, 'relu'))
model3.add(Dense(1, 'linear'))
model3.summary()


#df.drop(['holiday', 'weekend', 'dbh','dah','offer' ], axis=1)


plt.figure(figsize=(16,8))
plt.title('Sales History')
plt.plot(df['day'])
plt.xlabel('year', fontsize=16)
plt.ylabel('Price USD ($)', fontsize=18)
plt.show()


data = df.filter(["quantity"])
dataset = data.values
training_data_len= math.ceil(len(dataset)*.8)
training_data_len


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


train_data = scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<=60:
    print(x_train)
    print(y_train)
    print()


x_train, y_train = np.array(x_train), np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(25))

model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x_train, y_train, batch_size=1, epochs=20)

test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt( np.mean(predictions - y_test )**2)
rmse

"""
train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('quantity', fontsize=18)
plt.plot(train['quantity'])
plt.plot(valid[['quantity', 'predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show
"""

#valid
#print(df.loc[[99]])