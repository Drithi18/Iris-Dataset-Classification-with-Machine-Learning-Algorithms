#!/usr/bin/env python
# coding: utf-8

# In[58]:


# Project 2

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
print(sklearn.__version__)

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)


# In[59]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X = iris.data[:, 2:] # petal length and width   # <--------- modify here --------
print(type(iris.data))
print(iris.data.shape)
print(iris.data.ndim)
print(iris.data.size)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size = 0.3, random_state=42)
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_train, y_train)

rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(X_train, y_train)

#----------------------
# add naive bayes classifier and SVM here

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

#SVM

svm_clf = SVC(random_state = 42)
svm_clf.fit(X_train, y_train)

#----------------------
# modify models to include additional classifer: naive bayes
models = [("DT", tree_clf), ("RF", rnd_clf),("NB",gaussian)]


# modify models further to add additional classifier: SVM
models = [("DT", tree_clf), ("RF", rnd_clf),("NB",gaussian),("SVM",svm_clf)]

unsorted_scores = [(name, cross_val_score(model, X_train, y_train, cv=10).mean()) for name, model in models]
scores = sorted(unsorted_scores, key=lambda x: x[1])
print(scores)  


# In[60]:


from sklearn.metrics import classification_report, accuracy_score

#----------------------
# use the best classifier to run the test data

# Automate to find the best classifier and make it run the prediction,

model = {"DT" : tree_clf, "NB" : gaussian , "SVM" :svm_clf, "RF" :rnd_clf }
acc_list = []
for score in scores:
    acc_list.append(score[1])
    
#print(acc_list)
print('THE best Classifier : ',scores[acc_list.index(max(acc_list))][0])

y_pred = model[scores[acc_list.index(max(acc_list))][0]].predict(X_test)

print(accuracy_score(y_test,y_pred))


# In[61]:


# Test accuracy

y_pred_dt = tree_clf.predict(X_test)
print('tree : ' , accuracy_score(y_test,y_pred_dt))

y_pred_rf = rnd_clf.predict(X_test)
print('random forest : ', accuracy_score(y_test,y_pred_rf))


y_pred_nb = gaussian.predict(X_test)
print('naive bayes : ', accuracy_score(y_test,y_pred_nb))


# In[62]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()


# In[63]:


print(classification_report(y_test, y_pred, target_names=iris.target_names))


# In[ ]:




