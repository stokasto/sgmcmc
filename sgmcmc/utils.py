import numpy as np
import theano
import theano.tensor as T

def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None, broadcastable=None):
    if broadcastable:
        return theano.shared(np.asarray(X, dtype=dtype), name=name, broadcastable=broadcastable)
    else:
        return theano.shared(np.asarray(X, dtype=dtype), name=name)
