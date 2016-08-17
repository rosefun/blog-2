#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, linear_model, linear_model

def hingle(x):
    y = np.maximum(np.zeros(x.shape), 1-x)
    return y

x = np.arange(-10, 10, 0.1)
y_log = np.log(np.exp(0-x) + 1)
y_hingle = hingle(x)
fig, ax = plt.subplots(1)
ax.plot(x, y_log, label='LR : $\log(1+e^{-x})$', color='blue')
ax.plot(x, y_hingle, label='SVM : $\max(0,1-x)$', color='red')
ax.set_title('LR and SVM loss comparision')
ax.legend(loc='upper right')
ax.set_xlabel('$y(w^Tx+b)$')
ax.set_ylabel('')
ax.grid()

plt.savefig('loss_log_vs_hinge.png')

#plt.show()
