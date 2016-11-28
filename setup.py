import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'numpy',
    'theano',
    'autograd'
    ]

setup(name='sgmcmc',
      version='0.0.1',
      description='Framework for stochastic gradient markov chain monte carlo methods',
      packages=find_packages(),
      install_requires=requires)

