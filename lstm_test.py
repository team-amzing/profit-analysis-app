import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl as qdl

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from functools import reduce

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# API key for Michael's account
API_KEY = "NszGhwY_Qh8Ubj1BWhVt"

# Declare Quandl API key
qdl.ApiConfig.api_key = API_KEY

# Get data from the quandl api, then only return the number of days asked for
oil_price       = qdl.get("EIA/PET_RWTC_D"         , returns="pandas")
oil_production  = qdl.get("JODI/OIL_CRPRKD_WORLD"  , returns="pandas").drop("Notes", axis = 1)
oil_imports     = qdl.get("JODI/OIL_CRIMKT_WORLD"  , returns="pandas").drop("Notes", axis = 1)
oil_exports     = qdl.get("JODI/OIL_CREXBK_WORLD"  , returns="pandas").drop("Notes", axis = 1)
oil_reserves    = qdl.get("BP/OIL_RESERVES_WRLD"   , returns="pandas")
oil_consumption = qdl.get("BP/OIL_CONSUM_WRLD"     , returns="pandas")

dfs = [oil_price, oil_production, oil_imports, oil_exports, oil_reserves, oil_consumption]

min_dates = []
for df in dfs:
	min_dates.append(df.index[0])

max_date = max(min_dates)

df_final = reduce(lambda left,right: pd.merge(left,right,on='Date', how='outer'), dfs)
df_final = df_final.sort_values(by=['Date'])
df_final = df_final.ffill(axis = 0) 
df_final = df_final[df_final.index >= max_date]

df_final['tvalue'] = df_final.index
df_final['delta'] = (df_final['tvalue'].shift(-1)-df_final['tvalue']).fillna(0)
df_final['delta'] = df_final['delta'].dt.days

df_final = df_final.drop('tvalue', axis = 1)

values  = df_final.values

# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
maximum = values.max(axis = 0)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict

reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values

train = values[:len(values) - 1, :]
test = values[len(values) - 1:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

plt.figure()
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

saved_x = test_X
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(f'Test RMSE: {rmse:.3f}')

print(values[:,0][-10:] * maximum[0])

for days_ahead in range(5):
	example = saved_x
	example[0][0][6] = days_ahead/maximum[6]

	print(model.predict(example)*maximum[0])

reframed.drop(reframed.columns[[6]], axis=1, inplace=True)
reframed.columns = ["Price", "World Production", "World Imports", "World Exports", "World Reserves", "World Consumption", "Prediction"]
reframed.plot()
plt.savefig("predict.png")