import itertools
import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from collections import deque

import lasagne
from lasagne.layers import InputLayer, DenseLayer

from sgmcmc.utils import sharedX, floatX, shuffle
from sgmcmc.bnn.model import HMCBNN
from sgmcmc.bnn.lasagne_layers import AppendLayer

np.random.seed(8)
# generate some data
samples_x = 20
noise_scale = 0.004
X = np.random.uniform(-4, 4, size=(samples_x, 1)).astype(theano.config.floatX)
y = np.sinc(X) + np.random.normal(scale=noise_scale, size=(samples_x, 1)).astype(theano.config.floatX)
y = y.reshape(-1, 1)

Xtest = np.random.uniform(-6, 6, size=(200, 1)).astype(theano.config.floatX)
ytest = np.sinc(Xtest)
ytest = ytest.reshape(-1, 1)

# make sure outputs are zero mean and unit variance
# my code currently assumes that the targets are from
# a normal distribution and hence not doing this would
# lead to strange behaviour
mx = np.mean(X)
X -= mx
sx = np.std(X)
X /= sx
my = np.mean(y)
y -= my
sy = np.std(y)
y /= sy

Xtest -= mx
Xtest /= sx
ytest -= my
ytest /= sy


def get_net():
    l_in = InputLayer(shape=(None, 1))
    l_hid1 = DenseLayer(
        l_in, num_units=50,
        W = lasagne.init.HeNormal(),
        b = lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )
    l_hid2 = DenseLayer(
        l_hid1, num_units=50,
        W = lasagne.init.HeNormal(),
        b = lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )
    l_out = DenseLayer(
        l_hid2, num_units=1,
        W = lasagne.init.HeNormal(),
        b = lasagne.init.Constant(0.),
        nonlinearity=None)
    l_out = AppendLayer(l_out, num_units=1, b=lasagne.init.Constant(np.log(1e-3)))
    return l_out


burn_in = 1000
n_samples = 10000

model = HMCBNN(get_net, capture_every=50, burn_in=burn_in)

model.train(X, y, n_samples, epsilon=1e-2)

# sort from left to right
ys, var = model.sample_predictions(Xtest)
print(ys.shape, var.shape)
mean_pred = np.mean(ys, axis=0) #.reshape(-1)
std_pred  = np.std(ys, axis=0) #.reshape(-1)

v = np.mean(ys ** 2 + var, axis=0) - mean_pred ** 2

idx = np.argsort(Xtest, axis=0).reshape(-1)
Xtest = Xtest[idx]
ytest = ytest[idx]
mean_pred = mean_pred[idx]
std_pred = std_pred[idx]
v = v[idx]

plt.plot(X, y, 'x', color='blue')
plt.plot(Xtest, ytest, '--', color='blue', label='sinc(x)')
plt.plot(Xtest, mean_pred.reshape(-1), '-', color='red', label='Adaptive SGHMC')
pos = (mean_pred + 2*std_pred).reshape(-1)
neg = (mean_pred - 2*std_pred).reshape(-1)

pos2 = (mean_pred + 2*np.sqrt(v)).reshape(-1)
neg2 = (mean_pred - 2*np.sqrt(v)).reshape(-1)
plt.fill_between(Xtest.reshape(-1), pos2, neg2, color='red', alpha=0.2)
plt.ylim([-2, 2])
plt.xlim([np.min(Xtest), np.max(Xtest)])
plt.legend()
plt.title("A fit of the sinc function using SGHMC")

plt.savefig('sinc_sghmc.pdf')
plt.show()
