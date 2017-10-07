import numpy as np
from data_utils import DataUtils as du
import torch
from torch.autograd import Variable

TOTAL_DATASET_SIZE = 10887
epochs = 50000
train_data_size = 9600
batch_size = 64
steps_in_epoch = train_data_size//batch_size
layer_dims = {"in": 13, "fc1": 10, "fc2": 10, "fc3": 10,"fc4": 10, "out": 1}

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(layer_dims["in"], layer_dims["fc1"])
        self.tan1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(layer_dims['fc1'], layer_dims['fc2'])
        self.tan2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(layer_dims['fc2'], layer_dims['fc3'])
        self.tan3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(layer_dims['fc3'], layer_dims['fc4'])
        self.tan4 = torch.nn.Tanh()
        self.fc5 = torch.nn.Linear(layer_dims['fc4'], layer_dims['out'])
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        out = self.fc1(x)
        out = self.tan1(out)
        out = self.fc2(out)
        out = self.tan2(out)
        out = self.fc3(out)
        out = self.tan3(out)
        out = self.fc4(out)
        out = self.tan4(out)
        out = self.fc5(out)
        out = self.relu(out)
        return out

def rmsle(y_pred,y_true):
    log_err = torch.log(y_pred + 1) - torch.log(y_true + 1)
    squared_le = torch.pow(log_err,2)
    mean_sle = torch.mean(squared_le)
    root_msle = torch.sqrt(mean_sle)
    return (root_msle)

#Loading preprocessed dataframe. Full feature engineering, normalization, randomization.
datasetX,datasetY = du.get_processed_df('data/train.csv')

#Dividing data to train and val datasets, conversion from DF to numpyarray
X_train = np.array(datasetX[:train_data_size])
Y_train = np.array(datasetY[:train_data_size])
X_val = np.array(datasetX[train_data_size:])
Y_val = np.array(datasetY[train_data_size:])

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
X_train = Variable(torch.cuda.FloatTensor(X_train).cuda())
Y_train = Variable(torch.cuda.FloatTensor(Y_train).cuda())
X_train_batch = Variable(torch.randn(batch_size, 13).cuda())
Y_train_batch = Variable(torch.randn(batch_size).cuda())
X_val = Variable(torch.cuda.FloatTensor(X_val).cuda())
Y_val = Variable(torch.cuda.FloatTensor(Y_val).cuda())

# Use the nn package to define our net and loss dunction.
net = Network()
net.cuda()

optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)

mse = torch.nn.MSELoss()

for epoch in range(epochs):
    batch_ind = 0
    pred_val = net(X_val)
    val_loss = rmsle(pred_val, Y_val).data[0]
    pred_train = net(X_train)
    train_loss = rmsle(pred_train, Y_train).data[0]
    print("Epoch",epoch,": val loss", val_loss,", train loss:",train_loss)

    for step in range(steps_in_epoch):
        # Forward pass: compute predicted y by passing x to the net.
        X_train_batch = Variable(X_train.data[batch_ind:batch_ind + batch_size])
        Y_train_batch = Variable(Y_train.data[batch_ind:batch_ind + batch_size])
        y_pred = net(X_train_batch)

        # Compute and print loss.
        loss = mse(y_pred, Y_train_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        batch_ind += batch_size