# -*- codeing = utf-8 -*-
# @Time : 2021/05/13 0:38
# @Author : 217703 ZHANG WENXUAN
# @File : Chainer-MNISTClassifier_base.py
# @Software : PyCharm

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pandas as pd


class NeuralNet(chainer.Chain):
    def __init__(self):
        super(NeuralNet, self).__init__()
        with self.init_scope():
            self.layer1 = L.Convolution2D(None, 32, ksize=5)
            self.layer2 = L.Linear(None, 32)
            self.layer3 = L.Linear(None, 10)

    def __call__(self, x):
        x = self.layer1(F.relu(x))
        x = self.layer2(F.relu(x))
        x = self.layer3(F.relu(x))
        return x


df = pd.read_csv('../input/train.csv')
X = df[df.columns[1:]].astype(np.float32).values
Y = df[df.columns[0]].values

nn = NeuralNet()
model = L.Classifier(nn)

train_iter = chainer.iterators.SerialIterator([(X[i].reshape(1, 28, 28), Y[i]) for i in range(len(X))], 200,
                                              shuffle=True)
optimizer = chainer.optimizers.AdaDelta()
optimizer.setup(model)
updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = chainer.training.Trainer(updater, (5, 'epoch'), out="result")
trainer.extend(chainer.training.extensions.LogReport())
trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
trainer.run()

df = pd.read_csv('../input/test.csv')
df.head()
X_val = df.astype(np.float32).values
X_val = [X_val[i].reshape(1, 28, 28) for i in range(len(X_val))]
result = nn(np.array(X_val, dtype=np.float32))
result = [np.argmax(x) for x in result.data]
df = pd.DataFrame({'ImageId': range(1, len(result) + 1), 'Label': result})
df.to_csv('submission.csv', index=False)
