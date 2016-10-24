import itertools
import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import cPickle
from collections import deque

import lasagne
from lasagne.layers import InputLayer, DenseLayer

from sgmcmc.theano_mcmc import SGLDSampler, SGHMCSampler
from sgmcmc.utils import sharedX, floatX, shuffle
from sgmcmc.bnn.priors import WeightPrior, LogVariancePrior
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


def log_like(f_net, X, Y, n_examples, weight_prior, variance_prior):
    f_out = lasagne.layers.get_output(f_net, X)
    f_mean = f_out[:, 0].reshape((-1, 1))
    f_log_var = f_out[:, 1].reshape((-1, 1))
    f_var_inv = 1. / (T.exp(f_log_var) + 1e-16)
    MSE = T.square(Y - f_mean)
    log_like = T.sum(T.sum(-MSE * (0.5*f_var_inv) - 0.5*f_log_var, axis=1))
    # scale by batch size to make this work nicely with the updaters above
    log_like /= T.cast(X.shape[0], theano.config.floatX)
    #priors, scale these by dataset size for the same reason
    # prior for the variance
    tn_examples = T.cast(n_examples, theano.config.floatX) 
    log_like += variance_prior.log_like(f_log_var, n_examples) / tn_examples
    # prior for the weights
    params = lasagne.layers.get_all_params(f_net, trainable=True)
    log_like += weight_prior.log_like(params) / tn_examples
    return log_like, T.mean(MSE)

def sample_predictions(out_fun, f_net, mcmc_samples, X):
    y = []
    var = []
    for sample in mcmc_samples:
        lasagne.layers.set_all_param_values(f_net, sample)
        f_out = out_fun(X)
        y.append(f_out[:, 0])
        var.append(np.exp(f_out[:, 1]) + 1e-16)
                
    return np.asarray(y), np.asarray(var)


burn_in = 3000
n_samples = 30000
bsize = 20
n_examples = X.shape[0]
variance_prior = LogVariancePrior(1e-4, prior_out_std_prec=0.01)
weight_prior = WeightPrior(alpha=1., beta=1.)

net = get_net()
Xt = T.matrix()
Yt = T.matrix()
nll,mse = log_like(net, Xt, Yt, n_examples, weight_prior, variance_prior)
params = lasagne.layers.get_all_params(net, trainable=True)
sampler = SGHMCSampler(precondition=True, ignore_burn_in=True)
sampler.prepare_updates(-nll, params, np.sqrt(1e-4), mdecay=0.05, scale_grad=n_examples, inputs=[Xt, Yt])
compute_err = theano.function([Xt, Yt], [mse,nll])
predict = theano.function([Xt], lasagne.layers.get_output(net, Xt))

print("Starting sampling")
X, y = shuffle(X, y)
X = floatX(X)
y = floatX(y)
samples = deque(maxlen=100)
for i in range(n_samples):
    if X.shape[0] == bsize:
        start = 0
    else:
        start = (i * bsize) % (X.shape[0] - bsize)
    xmb = floatX(X[start:start+bsize])
    ymb = floatX(y[start:start+bsize]).reshape((-1, 1))
    _,nll = sampler.step(xmb, ymb)
    if i % 1000 == 0:
        total_err, total_nll = compute_err(floatX(X), floatX(y).reshape(-1,1))
        print("{}/{} : NLL = {} TOTAL={} ERR = {}".format(i, n_samples, nll,total_nll,total_err))
    if i > 10000 and i % 200 == 0:
        samples.append(lasagne.layers.get_all_param_values(net))

# sort from left to right
ys, var = sample_predictions(predict, net, samples, Xtest)
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
with open('sinc_sghmc_data.pkl', 'wb') as f:
    cPickle.dump({'X' : X, 'y' : y, 'Xtest' : Xtest, 'ytest' : ytest, 'mean_pred' : mean_pred, 'std_pred' : std_pred },f)
