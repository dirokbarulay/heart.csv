import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart.csv')

data.head()
data.sample(7)

data.info()
# <class 'pandas.core.frame.DataFrame'>

data.describe()
data['Oldpeak'].value_counts()


a = data.drop('Age', axis=1)
b = sns.pairplot(a, hue='Oldpeak', markers='+')
plt.show()


a = sns.violinplot(y='Oldpeak', x='RestingBP', data=data, inner='quartile')
plt.show()
a = sns.violinplot(y='Oldpeak', x='Cholesterol', data=data, inner='quartile')
plt.show()
a = sns.violinplot(y='Oldpeak', x='FastingBS', data=data, inner='quartile')
plt.show()



X = data.drop(['Age', 'Oldpeak'], axis=1)
Y = data['Oldpeak']
print(X.shape)
print(Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=5)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
