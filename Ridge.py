

import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt




data = np.load('linRegData.npy')




x = data[:,0]
x = x.reshape(-1,1)

y = data[:,1]
p = PolynomialFeatures(degree = 15)
x_ = p.fit_transform(x)

kf = KFold(n_splits = 10,shuffle = True)
alpha = [0.01, 0.05,0.1,0.5,1.0,5,10]
train_error = list()
val_error = list()




for a in alpha:
    clf = Ridge(alpha=a)
    val_temp = list()
    for train_index, test_index in kf.split(x_):
        clf.fit(x_[train_index],y[train_index])
        clf.score(x_[test_index],y[test_index])
        val_temp.append(1-clf.score(x_[test_index],y[test_index]))
    val_error.append(np.sum(val_temp)/(len(val_temp)))
    clf.fit(x_,y)
    train_error.append(1-clf.score(x_,y))


    #val_error.append(sum(val_temp)/float(len(val_temp)))

plt.plot(alpha,train_error)


plt.show()









