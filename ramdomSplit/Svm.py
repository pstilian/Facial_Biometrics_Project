# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 00:39:31 2021

@author: 30510
"""

from time import time
import logging
import sizeChange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import performance_plots
from os.path import join, exists, isdir
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB

"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"
"this code will change the original size of you pictues,you should use a copy of your pictures to try it !!!!!"

'''the voting classifier combined KNeighbors, RandomForest and svm''' 

print(__doc__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

'''path1 is the svm folder's directory'''
'''put your images in lfw_funneled folder, it's in lfw_home folder'''


path1 = 'F:\MobileBiometrics\Project1\svm' 
path2 = join(path1, "lfw_home\lfw_funneled")

''' it might take 15~20 minutes to change the size of pics'''
sizeChange.main2(path2)


lfw_people = fetch_lfw_people(data_home=path1,resize=0.4)

n_samples, h, w = lfw_people.images.shape



X = lfw_people.data
n_features = X.shape[1]


y = lfw_people.target
target_names = lfw_people.target_names 
n_classes = target_names.shape[0]   

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


n_components = 80

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', 
          whiten=True).fit(X_train)                       
print("done in %0.3fs" % (time() - t0))                  

eigenfaces = pca.components_.reshape((n_components, h, w))  

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)   
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
print("Fitting the classifier to the training set")

t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],   
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)

print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:") 
print(clf.best_estimator_)
print("Predicting people's names on the test set")

t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))  
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

n_splits=5

def voting_test():
    
    k_range = list(range(1,31))
    weight_options = ["uniform", "distance"]
    param_knn = dict(n_neighbors = k_range, weights = weight_options) 
    kn_clf = GridSearchCV(KNeighborsClassifier(), param_knn, cv = 5, scoring = 'accuracy')
    kn_clf.fit(X_train_pca, y_train)
    y_pred = kn_clf.predict(X_test_pca)
    print('KNeighborsClassifier ', accuracy_score(y_test, y_pred))
    
    #param_rf = {'n_estimators': [200, 700], 'max_features': ['auto', 'sqrt', 'log2']}
    #rf_clf = GridSearchCV(RandomForestClassifier(), param_rf, cv= 5)
    rf_clf = RandomForestClassifier(n_estimators=50, random_state=1)
    rf_clf.fit(X_train_pca, y_train)
    y_pred = rf_clf.predict(X_test_pca)
    print('RandomForestClassifier ', accuracy_score(y_test, y_pred))
    
    svm_clf = GridSearchCV(SVC(kernel='rbf',probability=True),
                   param_grid, cv=5)
    svm_clf.fit(X_train_pca, y_train)
    y_pred = svm_clf.predict(X_test_pca)
    print('SvmClassifier ', accuracy_score(y_test, y_pred))
    
    voting_clf = VotingClassifier(
            estimators = [('kn',kn_clf),('rf',rf_clf),('svm',svm_clf)],voting='hard')
    voting_clf.fit(X_train_pca, y_train)
    
    y_pred = voting_clf.predict(X_test_pca)
    print('VotingClassifier(hard) ', accuracy_score(y_test, y_pred))
    
    
    
    voting_clf2 = VotingClassifier(
            estimators = [('kn',kn_clf),('rf',rf_clf),('svm',svm_clf)],voting='soft')
    voting_clf2.fit(X_train_pca, y_train)
    
    
    matching_scores = voting_clf2.predict_proba(X_test_pca)
    gen_scores = []
    imp_scores = []
    classes = voting_clf2.classes_
    matching_scores = pd.DataFrame(matching_scores, columns=classes)
    for i in range(len(y_test)):    
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
    performance_plots.performance(gen_scores, imp_scores, 'VotingClassifier', 500)
    
    
    y_pred = voting_clf2.predict(X_test_pca)
    print('VotingClassifier(soft) ', accuracy_score(y_test, y_pred))

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(2.2 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)


eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

voting_test()


