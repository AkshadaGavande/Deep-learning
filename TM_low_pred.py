

"""
    Share-Tata motors
    Current Low prediction using PH+PO+PC+Volume
"""
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error


dataset = pandas.read_csv('TM_low.csv')

dataset = dataset.drop(['Date','Adj Close'], axis=1)
dataset.isnull().sum()
dataset.dropna(inplace=True)
#iloc is used for index where loc is used for label
data =dataset.drop(['Low'], axis = 1)
target = dataset['Low']
#convert dataframe in numpy array
data=data.values
target =target.values
data = data.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

#module for training and splitting
X_train=data[0:1225:]
X_test=data[1225:,:]
y_train=target[0:1225]
y_test=target[1225:,]

#converting the shape in the way machine will take for training
X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))

model = Sequential()
#64 neurons will be used at this layer
model.add(LSTM(64, input_shape=(1,4)))
#output layer contains 1 neuron to predict the output
model.add(Dense(1))
#as the data is continous, hence loss function is mean_squared_error
model.compile(loss='mean_squared_error', optimizer='sgd')
#data is trained here
model.fit(X_train, y_train, epochs=1500, batch_size=10, verbose=1)
#prediction using X_test
Predict = model.predict(X_test)

#testScore = math.sqrt(mean_squared_error(y_test, Predict))
#print('Test Score: %.2f RMSE' % (testScore))

plt.plot(y_test,color="r")
plt.plot(Predict,color="b")
#plt.legend(['original value','predicted value'],loc='upper right')
plt.show()

#---------------------------------------------



