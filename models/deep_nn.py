import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import sys
import copy
import matplotlib.pyplot as plt


class Net(nn.Module):
    # define nn
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)
        # self.fc2 = nn.Linear(num_features * 2, num_features * 2)
        # self.fc3 = nn.Linear(num_features * 2, num_features * 2)
        # self.fc4 = nn.Linear(num_features * 2, num_features * 2)
        # self.fc3 = nn.Linear(100, 100)
        # nn.Dropout(0.5)
        # self.fc3 = nn.Linear(1000,1000)
        # self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(50, 2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        # X = F.relu(self.fc2(X))
        # X = F.relu(self.fc3(X))
        # X = F.relu(self.fc4(X))
        X = self.fc5(X)
        # X = self.softmax(X)

        return X

# Usage python deep_nn.py <dataset csv filename> <learning rate> <momentum>
epochs = int(sys.argv[2])

dataset = pd.read_csv(sys.argv[1])
# dataset = dataset.drop(['Filename'], axis=1)
# dataset = dataset.drop(['Super_Nasty_Sig_Count'], axis=1)
# dataset = dataset.drop(['CloudflareBypass'], axis=1)
# dataset = dataset.drop(['SuspiciousEncoding'], axis=1)

dataset_norm = dataset.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# train_X, test_X, train_y, test_y = train_test_split(dataset_norm[dataset_norm.columns[0:4]].values,
#                                                     dataset_norm.Label.values.astype(int), test_size=0.3)

train_X, test_X, train_y, test_y = train_test_split(dataset_norm[dataset_norm.columns[0:-1]].values,
                                                    dataset_norm[dataset_norm.columns[dataset_norm.shape[1]-1]].values, test_size=0.3)

print(train_X.shape)
print(test_X.shape)
print(np.sum(train_y))
print(np.sum(test_y))
print(train_X[0], train_y[0])


# exit()

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

print(dataset.shape[1]-1)
net = Net(dataset.shape[1]-1)

print(net)

criterion = nn.CrossEntropyLoss()# cross entropy loss

# optimizer = torch.optim.SGD(net.parameters(), lr=float(sys.argv[2]), momentum=float(sys.argv[3]))
optimizer = torch.optim.Adam(net.parameters())

start_time = time.time()
best_loss = np.inf
no_improve = 0
epoch = 0
running_losses = []

for epoch in range(epochs):
# while no_improve < 100:
    epoch+=1
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if loss.data[0] < best_loss:
        best_loss = copy.deepcopy(loss.data[0])
        torch.save(net.state_dict(), 'checkpoint.pt')

    running_losses.append(loss.data[0])
        
    
    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data[0])

end_time = time.time()
print("Training Time:", end_time-start_time)

plt.plot(range(epochs), running_losses)
plt.show()

print(best_loss)
net.load_state_dict(torch.load('checkpoint.pt'))

predict_out = net(test_X)

_, predict_y = torch.max(predict_out, 1)

print('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

print('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
print('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
print('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
print('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))