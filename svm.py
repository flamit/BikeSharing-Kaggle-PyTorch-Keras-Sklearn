from sklearn.svm import SVR
import pandas as pd
import numpy as np
from data_utils import DataUtils as du
TOTAL_DATASET_SIZE = 10887

HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

def norm_arr(array):
    return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

def get_train_error(predictions_train,labels_train):
    total_error = 0
    for y_pred, label in zip(predictions_train, labels_train):
        total_error = total_error + (np.log(y_pred + 1) - np.log(label + 1)) ** 2
    return np.sqrt(total_error / len(predictions_train))

#Reading datasets
df = pd.read_csv('data/train.csv')
df_to_predict = pd.read_csv('data/test.csv')

#Removing Outliners fom dataset
#df_without_outliners = df[np.abs(df["count"]-df["count"].mean())<=(3*df["count"].std())]

#Adding continous time as variable because of the increasing amount of bikes over time
df['cont_time'] = df.datetime.apply(du.datetime_to_total_hours)
df_to_predict['cont_time'] = df_to_predict.datetime.apply(du.datetime_to_total_hours)

#Adding hour temporarily
df['hour'] = df.datetime.apply(du.get_hour)
df_to_predict['hour'] = df_to_predict.datetime.apply(du.get_hour)

#Little refactor of humidity to make it easier to learn
du.humidity_impact = np.array(df.groupby('humidity')['count'].mean())

# df['humidity'] = df.humidity.apply(du.get_humidity_impact)
# df_to_predict['humidity'] = df_to_predict.humidity.apply(du.get_humidity_impact)

#MONTH
df['month'] = df.datetime.apply(du.get_month)
#df_to_predict['month'] = df_to_predict.datetime.apply(du.get_month)

#Getting month impact, which tells us how good is the month for bikes, far better than 'season' and is easier to learn than pure month value
du.months_impact = np.array(df.groupby('month')['count'].mean())
df['month_impact'] = df.datetime.apply(du.get_month_impact)
df_to_predict['month_impact'] = df_to_predict.datetime.apply(du.get_month_impact)

#Year
df['year'] = df.datetime.apply(du.get_year)
df_to_predict['year'] = df_to_predict.datetime.apply(du.get_year)

df['day_of_week'] = df.datetime.apply(du.get_day_of_week)
df_to_predict['day_of_week'] = df_to_predict.datetime.apply(du.get_day_of_week)

#DAYS OF WEEK REG/CAS
du.days_of_week_reg = np.array(df.groupby('day_of_week')['registered'].mean())
du.days_of_week_cas = np.array(df.groupby('day_of_week')['casual'].mean())

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
#datasetX.loc[df.day_of_week != 5, 'hours_sat']

#Hour impact for registered and casual
df['hour_reg'] = df.datetime.apply(du.get_hour_registered)
df['hour_cas'] = df.datetime.apply(du.get_hour_casual)
df_to_predict['hour_reg'] = df_to_predict.datetime.apply(du.get_hour_registered)
df_to_predict['hour_cas'] = df_to_predict.datetime.apply(du.get_hour_casual)

#Data randomization(shuffling)
df = df.sample(frac=1).reset_index(drop=True)

#Spitting data into input features and labels
datasetY = df.ix[:,'count']
datasetX = df.drop(['casual','registered','count','datetime','windspeed','atemp','season','month'],1)
datasetX_pred = df_to_predict.drop(['datetime','windspeed','atemp','season'],1)

#Normalizing inputs
datasetX = (datasetX - datasetX.min() - (datasetX.max() - datasetX.min())/2) / ((datasetX.max() - datasetX.min())/2)
datasetX_pred = (datasetX_pred - datasetX_pred.min() - (datasetX_pred.max() - datasetX_pred.min())/2) / ((datasetX_pred.max() - datasetX_pred.min())/2)

# datasetX = datasetX.drop(['day_of_week'],1)
# datasetX_pred = datasetX_pred.drop(['day_of_week'],1)
#datasetX.head(10)

print("Trainset:",datasetX.columns.values,datasetX.shape)
print("Testset:",datasetX_pred.columns.values,datasetX_pred.shape)

#Dividing the original train dataset into train/test set, whole set because keras provides spliting to cross-validation and train set
train_setX = datasetX.ix[:,:]
train_setY = datasetY.ix[:]
# train_setX = datasetX.ix[:TRAIN_SIZE-1,:]
# train_setY = datasetY.ix[:TRAIN_SIZE-1]

#Conversion from DF to numpyarray for Keras duncs
train_setX = np.array(train_setX)
train_setY = np.array(train_setY)

#Training our model

svr_lin = SVR(kernel='linear', C=1000)
svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.03)
svr_poly = SVR(kernel='poly', C=1000, degree=2, gamma=0.05)
classifiers = [svr_lin,svr_rbf,svr_poly]

for name,classifier in zip(["linear","gaussian","polynomial"],classifiers):
    classifier.fit(datasetX, datasetY)

    #Making predictions on train set and setting negative results to zero
    predictions_train = classifier.predict(datasetX)
    get_positive_vals = lambda x: x if x >=0 else 0
    predictions_train = [get_positive_vals(y) for y in predictions_train]

    #Calculating error on train set
    labels_train = np.array(df.ix[:,'count'])
    train_error = get_train_error(predictions_train,labels_train)
    print ("Error for",name,"kernel:",train_error)

    #Making predictions on test set and setting negative results to zero
    predictions_test = classifier.predict(datasetX_pred)
    predictions_test_final = [get_positive_vals(y) for y in predictions_test]

    #Saving predictions
    #np.savetxt("svm_predictions.csv", predictions_test_final, delimiter=",")
