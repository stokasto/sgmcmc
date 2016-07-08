import autograd
import autograd.numpy as np
import autograd.numpy.random as npr

from sgmcmc.autograd_mcmc import SGLDSampler
import matplotlib.pyplot as plt


Xsize = 2000

def neg_log_like(theta, inputs):
    return np.mean((inputs - theta[0]) ** 2 / (2. * np.exp(theta[1]) + 1e-10) + theta[1]/2.)

def joint_neg_log_like(theta, inputs):
    global Xsize
    prior = np.sum(np.square(theta)) * 1
    nll = Xsize * neg_log_like(theta, inputs)
    return nll + prior

epsilon = 0.01
A = 1
theta = npr.normal(size=(2,))

sampler = SGLDSampler(precondition=True, noise_correction=False)
updates = sampler.prepare_updates(joint_neg_log_like, theta, epsilon, A=A)

# draw data
std = 0.8
mean = 1.4
x = np.random.normal(size=(Xsize, 1)) * std + mean

bsize = 20
n_samples = 8 * 10**4
samples = []
for i in range(n_samples):
    start = (i * bsize) % (x.shape[0] - bsize)
    xmb = np.copy(x[start:start+bsize])
    sample = sampler.step(xmb)

    samples.append(np.random.randn() * np.sqrt(np.exp(sample[1]) + 1e-16) + sample[0])

true_step = 0.001
xs = np.arange(-6, 6, true_step)
nlls = np.zeros_like(xs)
for i,x in enumerate(xs):
    nlls[i] = neg_log_like(np.array([mean, np.log(std**2)]), x)

# compute likelihood
lls = np.exp(-nlls)
# approximately compute z via euler integration
z = np.sum(lls) * true_step

# approximate density from the samples
step_sample = 0.1
xgrid = np.arange(-6, 6, step_sample) 
ygrid = np.asarray(np.histogram(samples, xgrid, density=True)[0])
plt.plot(xs, lls/z, color='red')
plt.plot(xgrid[:len(ygrid)], ygrid)
plt.figure()
ygrid2 = np.histogram(samples, 1500)[0]
plt.scatter(np.arange(len(ygrid)), ygrid)
#plt.scatter(samples, samples)
plt.show()

