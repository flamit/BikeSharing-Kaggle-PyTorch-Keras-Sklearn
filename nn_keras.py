import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import keras.backend as KB
from keras.layers import Dropout
from data_utils import DataUtils as du

TOTAL_DATASET_SIZE = 10887

def rmsle(y_true, y_pred):
    return KB.sqrt(KB.mean(KB.square(KB.log(y_pred+1) - KB.log(y_true+1)), axis=-1))

def norm_arr(array):
    return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

#Reading datasets
df = pd.read_csv('data/train.csv')
df_to_predict = pd.read_csv('data/test.csv')

#Adding continous time as variable because of the increasing amount of bikes over time
df['cont_time'] = df.datetime.apply(du.datetime_to_total_hours)
df_to_predict['cont_time'] = df_to_predict.datetime.apply(du.datetime_to_total_hours)

#Adding hour temporarily
df['hour'] = df.datetime.apply(du.get_hour)
df_to_predict['hour'] = df_to_predict.datetime.apply(du.get_hour)

#Little refactor of humidity to make it easier to learn
du.humidity_impact = np.array(df.groupby('humidity')['count'].mean())

#Month
df['month'] = df.datetime.apply(du.get_month)

#Getting month impact, which tells us how good is the month for bikes, far better than 'season' and is easier to learn than pure month value
du.months_impact = np.array(df.groupby('month')['count'].mean())
df['month_impact'] = df.datetime.apply(du.get_month_impact)
df_to_predict['month_impact'] = df_to_predict.datetime.apply(du.get_month_impact)

#Year
df['year'] = df.datetime.apply(du.get_year)
df_to_predict['year'] = df_to_predict.datetime.apply(du.get_year)

#Day of week
df['day_of_week'] = df.datetime.apply(du.get_day_of_week)
df_to_predict['day_of_week'] = df_to_predict.datetime.apply(du.get_day_of_week)

#DAY OF WEEK REG/CAS
du.days_of_week_reg = np.array(df.groupby('day_of_week')['registered'].mean())
du.days_of_week_cas = np.array(df.groupby('day_of_week')['casual'].mean())
df['day_of_week_reg'] = df.day_of_week.apply(du.get_day_of_week_reg)
df['day_of_week_cas'] = df.day_of_week.apply(du.get_day_of_week_cas)
df_to_predict['day_of_week_reg'] = df_to_predict.day_of_week.apply(du.get_day_of_week_reg)
df_to_predict['day_of_week_cas'] = df_to_predict.day_of_week.apply(du.get_day_of_week_cas)

#Hour impact array
du.hours_impact = np.array(df.groupby('hour')['count'].mean())

#Hour impact arrays for registered and casual
du.hours_cas = np.array(df.groupby('hour')['casual'].mean())
du.hours_reg = np.array(df.groupby('hour')['registered'].mean())

#Hour impact arrays for workingday, freeday, sat, sun
du.hours_workday = norm_arr(df.loc[df['workingday'] == 1].groupby('hour')['count'].mean())
du.hours_freeday = norm_arr(df.loc[(df['workingday'] == 0) & (df['day_of_week'] < 5)].groupby('hour')['count'].mean())
du.hours_sat = norm_arr(df.loc[df['day_of_week'] == 5].groupby('hour')['count'].mean())
du.hours_sun = norm_arr(df.loc[df['day_of_week'] == 6].groupby('hour')['count'].mean())

#Hour impact for registered and casual
df['hour_reg'] = df.datetime.apply(du.get_hour_registered)
df['hour_cas'] = df.datetime.apply(du.get_hour_casual)
df_to_predict['hour_reg'] = df_to_predict.datetime.apply(du.get_hour_registered)
df_to_predict['hour_cas'] = df_to_predict.datetime.apply(du.get_hour_casual)
#print(df.head(10))

#Data randomization(shuffling)
df = df.sample(frac=1).reset_index(drop=True)
#print(df.head(30))

#Spitting data into input features and labels
datasetY = df.ix[:,'casual':'count']
datasetX = df.drop(['casual','registered','count','datetime','windspeed','atemp','season','month'],1)
datasetX_pred = df_to_predict.drop(['datetime','windspeed','atemp','season'],1)
#print(datasetY.head(10))

#Normalizing inputs
datasetX = (datasetX - datasetX.min() - (datasetX.max() - datasetX.min())/2) / ((datasetX.max() - datasetX.min())/2)
datasetX_pred = (datasetX_pred - datasetX_pred.min() - (datasetX_pred.max() - datasetX_pred.min())/2) / ((datasetX_pred.max() - datasetX_pred.min())/2)

datasetX = datasetX.drop(['day_of_week'],1)
datasetX_pred = datasetX_pred.drop(['day_of_week'],1)

print("Features used:",datasetX.shape)
print("Final train set:",datasetX.head(10))

#Dividing the original train dataset into train/test set, whole set because keras provides spliting to cross-validation and train set
train_setX = datasetX
train_setY = datasetY

#Conversion from DF to numpyarray for Keras duncs
train_setX = np.array(train_setX)
train_setY = np.array(train_setY)

deep_layers_size = 10

#Defining our NN model
model = Sequential()

#Input and 1st deep layer
model.add(Dense(units=deep_layers_size, input_dim=13,kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation("tanh"))

#2nd deep layer
model.add(Dense(units=deep_layers_size,kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation("tanh"))

#3rd deep layer
model.add(Dense(units=deep_layers_size,kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation("tanh"))

#4th deep layer
model.add(Dense(units=deep_layers_size,kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation("tanh"))

#Output layer
model.add(Dense(units=3,kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Activation("relu"))


model.compile(loss=rmsle, optimizer='adam')

#Defining checkpoint and callbacks to save the best set of weights and limit printing
checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#Start training
history_callback = model.fit(train_setX, train_setY, epochs=0000, batch_size=50,validation_split=0.1,verbose=2,callbacks=callbacks_list)

#Recovering val_loss history and training loss history from callbacks to arrays
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]

np.savetxt("error_plot.csv", [loss_history,val_loss_history], delimiter=",")

#Loading weights
model.load_weights('best_weights.hdf5')

#Making predictionsand saving them to csv
predictions = model.predict(np.array(datasetX_pred))
predictions_count = predictions[:,-1]
np.savetxt("predictions.csv", predictions_count, delimiter=",")
np.savetxt("all_predictions.csv", predictions, delimiter=",")

#Plotting training loss and validation loss to control overfitting
plt.plot(loss_history)
plt.plot(val_loss_history)
plt.show()

