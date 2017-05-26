
# coding: utf-8

# In[1]:

import pandas as pd 
import numpy as np
import sklearn.preprocessing as sk_pre


import keras



# In[2]:

dataset = pd.read_csv('./data/QLD_all.csv', index_col=0)
dataset.columns = ['region', 'date', 'demand', 'price', 'type']
dataset = dataset.drop('type',axis=1)
#dataset.index = dataset.date


# In[3]:

dataset.head()


# In[4]:

# lets take difference for making stationary data
price_diff = dataset.price.diff()[1:]


# In[5]:

supervised_dataset = pd.DataFrame()
supervised_dataset['price'] = price_diff.shift()
supervised_dataset['label'] = price_diff
supervised_dataset = supervised_dataset.fillna(0,axis=1)


# In[6]:

supervised_dataset.head()


# In[7]:

X,y = supervised_dataset.price.values, supervised_dataset.label.values
scaler = sk_pre.MinMaxScaler(feature_range=(-1,1))
X  = X.reshape(X.shape[0],1)
X.shape


# In[8]:

scaler.fit(X)
X_scaled = scaler.transform(X)
y_scaled = scaler.transform(y.reshape(X.shape[0],X.shape[1]))


# In[9]:

# split train and test data
ts_size = 25000
X_train,X_test,y_train,y_test = X_scaled[0:ts_size],X_scaled[ts_size:],y_scaled[0:ts_size],y_scaled[ts_size:]
X_train = X_train.reshape(X_train.shape[0],1,1)
X_test = X_test.reshape(X_test.shape[0],1,1)
y_train = y_train.reshape(1,len(y_train))[0]
y_test = y_test.reshape(1,len(y_test))[0]


# In[10]:

X_train


# In[11]:

# fit an LSTM network to training data
def fit_lstm(X,y, batch_size, nb_epoch, neurons):
    print(X.shape)
    model = keras.models.Sequential()
    model.add(keras.layers.recurrent.LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        if i%100 == 0:
            print('Epoch #',i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=5, shuffle=False)
        model.reset_states()
    return model


# In[12]:

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# In[13]:

#lstm_model = fit_lstm(X_train,y_train,1,2,4)
X_train.shape
#y_train.shape
fit_lstm(X_train,y_train,1,1,1)


# (X_train.reshape(X_train.shape[0], 1, X_train.shape[1])).shape
# y_train.reshape(y_train.shape[0],)
# y_train.shape

# 
# model = keras.models.Sequential()
# model.add(keras.layers.recurrent.LSTM(1, batch_input_shape=(1, 1, 1), stateful=True))
# model.add(keras.layers.Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train.reshape(X_train.shape[0], 1, 1), y_train, epochs=5, batch_size=1, verbose=5, shuffle=False)
# model.reset_states()

# #fit the model
# lstm_model = fit_lstm(X_train,y_train,
#                      batch_size=1,
#                      nb_epoch=1,
#                      neurons=4)

# In[ ]:




# In[ ]:



