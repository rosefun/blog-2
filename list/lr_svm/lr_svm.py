#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, linear_model, linear_model

x_pos = np.random.uniform(3.8, 4.2, (10000, 2))
x_neg = np.random.uniform(-4.2, -3.8, (100, 2))

y_pos = np.full(10000, 0)
y_neg = np.full(100, 1)

x = np.concatenate((x_pos, x_neg), axis=0)
y = np.concatenate([y_pos, y_neg])

#svc = svm.SVC(kernel='linear', C=10000).fit(x, y)
hinglesgd = linear_model.SGDClassifier(loss = "hinge", penalty = "l2", shuffle = True, average = 10, alpha = 0.00001).fit(x, y)
logsgd = linear_model.SGDClassifier(loss = "log", penalty = "l2", shuffle = True, average = 10, alpha = 0.00001).fit(x, y)

# create a mesh to plot in
h = .02  # step size in the mesh
#x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
#y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
x_min, x_max = -6, 6
y_min, y_max = -6, 6
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['hingle SGD(SVM)',
          'Log SGD(LR)']

fig = plt.figure()
fig.suptitle("SVM and LR comparision")
for i, clf in enumerate((hinglesgd, logsgd)):
# Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    ax = fig.add_subplot(1, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    #plt.axes().set_aspect(1)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(titles[i])

plt.savefig('lr_svm.png')
#plt.show()
