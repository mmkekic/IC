#!/usr/bin/env python
import traceback
from sys       import argv
from importlib import import_module
import numpy
from invisible_cities.core.configure import configure
from invisible_cities.cities.detsim import detsim
numpy.random.seed(0)

conf = configure('detsim detsim.conf'.split())

from time import time
t0=time()
detsim(**conf)
print(time()-t0)
