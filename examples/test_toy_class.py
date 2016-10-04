import theano.tensor as T
import theano
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons

import lasagne
from lasagne.layers import DenseLayer, InputLayer
from sgmcmc.theano_mcmc import SGLDSampler
from sgmcmc.utils import sharedX, floatX, shuffle
import sacred

from collections import deque

ex = sacred.Experiment("toy_class_sgld")

@ex.config
def config():
    lrate = 1e-4
    n_samples = 5 * 10**4
    bsize = 100
    n_nets = 100

def mean_binary_crossent(out, Y):
    return T.mean(-Y * T.log(out + 1e-4) - (1.-Y) * T.log(1. - out + 1e-4))

def get_net():
    l_in = InputLayer(shape=(None, 2))
    l_hid1 = DenseLayer(
        l_in, num_units=50,
        W = lasagne.init.HeNormal(),
        b = lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.elu
    )
    l_hid2 = DenseLayer(
        l_hid1, num_units=50,
        W = lasagne.init.HeNormal(),
        b = lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.elu
    )
    l_out = DenseLayer(
        l_hid2, num_units=1,
        W = lasagne.init.HeNormal(),
        b = lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_out

def neg_log_like(net, X, Y, Xsize=1, wd=1):
    all_params = lasagne.layers.get_all_params(net, trainable=True)
    out = lasagne.layers.get_output(net, X)
    data_nll = Xsize * mean_binary_crossent(out, Y)
    prior_nll = 0.
    for p in all_params:
        prior_nll += T.sum(p**2) * 0. * wd
    U = data_nll + prior_nll
    return U, all_params

def class_error(net, X, Y):
    out = lasagne.layers.get_output(net, X)
    binarized = T.switch(out > 0.5, 1, 0)
    data_nll = mean_binary_crossent(out, Y)
    return T.mean(abs(out - Y)), data_nll


@ex.automain
def main(lrate, n_samples, bsize, n_nets):
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X, Y = shuffle(X, Y)
    X = scale(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
    net = get_net()
    sampler = SGLDSampler(precondition=True)
    all_params = lasagne.layers.get_all_params(net, trainable=True, A=1)

    Xt = T.matrix()
    Yt = T.matrix()
    U, params = neg_log_like(net, Xt, Yt, Xsize=X_train.shape[0])
    # we could also use these updates in our custom function
    # but instead we will use the sampler.step function below
    updates = sampler.prepare_updates(U, params, lrate, inputs=[Xt, Yt])
    err = class_error(net, Xt, Yt)
    compute_err = theano.function([Xt, Yt], err)
    predict = theano.function([Xt], lasagne.layers.get_output(net, Xt))

    print("Starting sampling")
    samples = deque(maxlen=n_nets)
    for i in range(n_samples):
        start = (i * bsize) % (X_train.shape[0] - bsize)
        xmb = floatX(X_train[start:start+bsize])
        ymb = floatX(Y_train[start:start+bsize]).reshape((-1, 1))
        _,nll = sampler.step(xmb, ymb)
        if i % 1000 == 0:
            total_err, total_nll = compute_err(floatX(X_train), floatX(Y_train).reshape(-1,1))
            print("{}/{} : NLL = {} TOTAL={} ERR = {}".format(i, n_samples, nll,total_nll,total_err))
        if i % 200 == 0:
            samples.append(lasagne.layers.get_all_param_values(net))

    # get predictions
    grid = np.mgrid[-3:3:100j,-3:3:100j]
    grid_2d = floatX(grid.reshape(2, -1).T)
    preds = np.zeros((grid_2d.shape[0], len(samples)))
    preds_test = np.zeros((X_test.shape[0], len(samples)))
    for i,sample in enumerate(samples):
        lasagne.layers.set_all_param_values(net, sample)
        preds[:, i]      = predict(grid_2d).reshape(-1)
        preds_test[:, i] = predict(floatX(X_test)).reshape(-1)

    mean_pred = np.mean(preds, axis=1)
    std_pred = np.std(preds, axis=1)

    mean_pred_test = np.mean(preds_test, axis=1)
    class_pred_test = mean_pred_test > 0.5
    std_pred_test = np.std(preds_test, axis=1)

    
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(grid[0], grid[1], mean_pred.reshape(100, 100), cmap=cmap, alpha=1.)
    ax.scatter(X_test[class_pred_test==0, 0], X_test[class_pred_test==0, 1])
    ax.scatter(X_test[class_pred_test==1, 0], X_test[class_pred_test==1, 1], color='r')
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y')
    cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(grid[0], grid[1], std_pred.reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[class_pred_test==0, 0], X_test[class_pred_test==0, 1])
    ax.scatter(X_test[class_pred_test==1, 0], X_test[class_pred_test==1, 1], color='r')
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
    cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');
    plt.show()
