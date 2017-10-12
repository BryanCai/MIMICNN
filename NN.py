import numpy as np
import lutorpy as lua
import sys
from sklearn.preprocessing import maxabs_scale
from sklearn.metrics import roc_auc_score

dataSize = "10000"
timeCut = "8h"

Cardio = np.genfromtxt("data/" + dataSize + "/" + timeCut + "/Cardio.csv", delimiter=',', skip_header=1)
Chem   = np.genfromtxt("data/" + dataSize + "/" + timeCut + "/Chemistries.csv", delimiter=',', skip_header=1)
Hema   = np.genfromtxt("data/" + dataSize + "/" + timeCut + "/Hematology.csv", delimiter=',', skip_header=1)
Misc   = np.genfromtxt("data/" + dataSize + "/" + timeCut + "/Misc.csv", delimiter=',', skip_header=1)
Urine  = np.genfromtxt("data/" + dataSize + "/" + timeCut + "/UrineIO.csv", delimiter=',', skip_header=1)
Vent   = np.genfromtxt("data/" + dataSize + "/" + timeCut + "/Ventilation.csv", delimiter=',', skip_header=1)

data = np.concatenate([Cardio, Chem[:,1:], Hema[:,1:], Misc[:,1:], Urine[:,1:], Vent[:,1:]], axis = 1)
maxabs_scale(data, axis = 0, copy = False)
np.random.shuffle(data)

#data[data[:,0] == 0, 0] = -1


valdata = data[:3000]
traindata = data[3000:]

require("nn")
require("torch")
require("optim")
require("xlua")

dispFreq = 1
nEpochs = 300
batchSize = 1024
lr = 1.0

STATUS_STR = 'Epoch %2d | train loss: %.3f | val loss: %.3f | val auc: %.2f%%'

model = nn.Sequential()\
        ._add(nn.Linear(data.shape[1] - 1, 500))\
        ._add(nn.ReLU())\
        ._add(nn.Linear(500, 100))\
        ._add(nn.ReLU())\
        ._add(nn.Dropout())\
        ._add(nn.Linear(100, 1))\
        ._add(nn.Sigmoid())

crit = nn.BCECriterion()

print(string.format("dataSize: %d, timeCut: %s, nEpochs: %d, batchSize: %d, lr: %.3f", dataSize, timeCut, nEpochs, batchSize, lr))
print model
print crit

trainSize = traindata.shape[0]
valSize = valdata.shape[0]
for i in range(nEpochs):
    trainLoss = 0
    trainBatches = 0
    model._training()
    np.random.shuffle(traindata)
    for n in range(int(np.ceil(trainSize/batchSize))):
        sys.stdout.write(string.format("training iteration: %d of %d" + " "*20 + "\r", n, int(np.ceil(trainSize/batchSize))))
        sys.stdout.flush()
        upper = np.minimum((n+1)*batchSize, trainSize)
        xt = torch.fromNumpyArray(traindata[n*batchSize:upper, 1:])
        yt = torch.fromNumpyArray(traindata[n*batchSize:upper, 0])

        model._forward(xt)
        trainLoss = trainLoss + crit._forward(model.output, yt)
        trainBatches = trainBatches + 1

        crit._backward(model.output, yt)
        model._zeroGradParameters()
        model._backward(xt, crit.gradInput)

        model._updateParameters(lr)

    model._evaluate()
    np.random.shuffle(valdata)
    xt = torch.fromNumpyArray(valdata[:,1:])
    yt = torch.fromNumpyArray(valdata[:, 0])

    model._forward(xt)
    valAUC = roc_auc_score(yt.asNumpyArray(), model.output.asNumpyArray().reshape(-1))
    valLoss = crit._forward(model.output, yt)

    if (i + 1)%50 == 0:
        lr /= 2.0
        print str(lr) + " "*20

    print(string.format(STATUS_STR, i, trainLoss/trainBatches, valLoss, valAUC*100))
