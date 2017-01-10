#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年01月08日 星期日 11时56分22秒
# last modified:

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import gamma as Gamma

def normal(x, m, s):
    p = 1./np.sqrt(2*np.pi*s**2)*np.exp(-(x-m)**2/(2.*s**2))
    return p

def beta(a, b, x):
    return Gamma(a+b)/(Gamma(a)*Gamma(b))*x**(a-1)*(1-x)**(b-1)

x = np.linspace(0.0, 1., 100)
lw = 2
plt.figure()
plt.subplot(231)
plt.plot(x, beta(1, 4, x), label='a=1, b=4', linewidth=lw)
plt.axis([0, 1, 0, 6])
plt.legend()

plt.subplot(232)
plt.plot(x, beta(3, 4, x), label='a=3, b=4', linewidth=lw)
plt.legend()

plt.subplot(233)
plt.plot(x, beta(4, 4, x), label='a=4, b=4', linewidth=lw)
plt.legend()

plt.subplot(234)
plt.plot(x, beta(6, 4, x), label='a=6, b=4', linewidth=lw)
plt.legend()

plt.subplot(235)
plt.plot(x, beta(8, 4, x), label='a=8, b=4', linewidth=lw)
plt.legend()

plt.subplot(236)
plt.plot(x, beta(10, 4, x), label='a=10, b=4', linewidth=lw)
plt.legend()

plt.show()
