from sklearn.svm import SVR
import numpy as np
from data_utils import DataUtils as du
from matplotlib import pyplot as plt

TOTAL_DATASET_SIZE = 10887
HOURS_IN_DAY = 24
START_YEAR = 2011
DAYS_IN_YEAR = 365
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_IN_YEAR = 12

TRAIN_SIZE = 9000

def norm_arr(array):
    return (array - array.min() - (array.max() - array.min()) / 2) / ((array.max() - array.min()) / 2)

def get_train_error(predictions_train,labels_train):
    total_error = 0
    for y_pred, label in zip(predictions_train, labels_train):
        total_error = total_error + (np.log(y_pred + 1) - np.log(label + 1)) ** 2
    return np.sqrt(total_error / len(predictions_train))

def rmsle(y_pred,y_true):
    #print(np.log(y_pred+1).shape,np.log(y_true+1).shape)
    log_err = (np.log(y_pred + 1) - np.log(y_true + 1))
    squared_le = np.power(log_err,2)
    mean_sle = np.mean(squared_le)
    root_msle = np.sqrt(mean_sle)
    return (root_msle)

if __name__ == '__main__':

    datasetX,datasetY,datasetX_pred = du.get_processed_df('data/train.csv','data/test.csv')

    #Conversion from DF to numpyarray for Keras duncs
    datasetX = np.array(datasetX)
    datasetY = np.array(datasetY)
    datasetX_pred = np.array(datasetX_pred)

    #Dividing the original train dataset into train/test set, whole set because keras provides spliting to cross-validation and train set
    X_train = datasetX[:TRAIN_SIZE]
    Y_train = datasetY[:TRAIN_SIZE]
    X_val = datasetX[TRAIN_SIZE:]
    Y_val = datasetY[TRAIN_SIZE:]

    print("Train set:",X_train.shape)
    print("Test set:",X_val.shape)

    #Training our model
    #svr_lin = SVR(kernel='linear', C=1000)
    #svr_poly = SVR(kernel='poly', C=1000, degree=2, gamma=0.5)

    max_gamma_val = 0.1
    min_gamma_val = 0.3
    max_c_val = 400
    min_c_val = 150
    steps = 30
    repeats = 2

    gamma_vals = np.linspace(max_gamma_val, min_gamma_val, steps)
    c_vals = np.linspace(max_c_val, min_c_val, steps)

    train_err_his = np.zeros(steps)
    val_err_his = np.zeros(steps)

    name = "Gaussian"

    for i,gamma_val in enumerate(gamma_vals):
        for j in range(repeats):
            if j ==0:
                X_train = datasetX[:TRAIN_SIZE]
                Y_train = datasetY[:TRAIN_SIZE]
                X_val = datasetX[TRAIN_SIZE:]
                Y_val = datasetY[TRAIN_SIZE:]
            else:
                X_train = datasetX[TRAIN_SIZE:]
                Y_train = datasetY[TRAIN_SIZE:]
                X_val = datasetX[:TRAIN_SIZE]
                Y_val = datasetY[:TRAIN_SIZE]
            classifier = SVR(kernel='rbf', C=325, gamma=gamma_val)
            classifier.fit(X_train, Y_train)

            #Making predictions on train set and setting negative results to zero
            predictions_train = classifier.predict(X_train)
            predictions_train = np.maximum(predictions_train, 0)
            train_error = rmsle(predictions_train,Y_train)
            train_err_his[i] += train_error

            predictions_val = classifier.predict(X_val)
            predictions_val = np.maximum(predictions_val, 0)
            val_error = rmsle(predictions_val,Y_val)
            val_err_his[i] += val_error
            #print (name,"kernel: Train error:",train_error,", Val error:",val_error)

    val_err_his = val_err_his / 2
    train_err_his = train_err_his / 2

    plt.plot(gamma_vals,train_err_his,label='train')
    plt.plot(gamma_vals,val_err_his,label='val')
    plt.xlabel('gamma', fontsize=16)
    plt.ylabel('error', fontsize=16)
    plt.legend()
    plt.show()


    #Saving predictions
    # Making predictions on test set and setting negative results to zero
    # predictions_test = classifier.predict(datasetX_pred)
    # predictions_test = np.maximum(predictions_test, 0)
    #np.savetxt("svm_"+name+"_predictions.csv", predictions_test, delimiter=",")
