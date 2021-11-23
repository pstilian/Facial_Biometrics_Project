import numpy as np
import pickle
import math
import pandas as pd
import performance_plots
from data_reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from haroun.losses import rmse
from haroun import Data, Model, ConvPool
import torch
import sys
import skimage.transform as tf

sys.path.append("../")
from deepfake_model import Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''run FaceCapture.py before run this file'''
'''run FaceCapture.py before run this file'''
'''run FaceCapture.py before run this file'''
'''The first VotingClassifier is voting='hard', the second one is voting='soft' '''

# checks each image in the dataset to determine if it is a fake
# if so, the image pixel values are zeroed out
def detect_deepfakes(data, AntiSpoofClassifier):
    batch = np.empty((0,64,64))
    with torch.no_grad():
        for img in data[0]:
            # print(img.values.shape)
            img.values = tf.resize(img.values, (64,64))
            batch = np.concatenate((batch,np.expand_dims(img.values,axis=0)))
        
        batch = torch.Tensor(batch,device=device).unsqueeze(1)
        batch = batch.expand(-1,3,-1,-1)
        result = AntiSpoofClassifier.net(batch)
        # TODO: iterate through predictions and zero out fakes



def main():
    data_reader = DataReader()  # reads the images files and converts them into numpy 2D arrays
    feature_extractor = FeatureExtractor()  # calculates the eigenfaces. Follows the fit->transform paradigm.
    #clf = GaussianNB()  # a naive bayes classifier where the individual variables are supposed to follow a gaussian distribution   
    kn_clf = KNeighborsClassifier()
    
    bl_clf = BernoulliNB()
    
    rf_clf = RandomForestClassifier()
    
    svm_clf = SVC(kernel='rbf',probability=True)
    
    voting_clf = VotingClassifier(
            estimators = [('kn',kn_clf),('bl',bl_clf),('rf',rf_clf),('svm',svm_clf)],voting='hard')
    
    voting_clf2 = VotingClassifier(
            estimators = [('kn',kn_clf),('bl',bl_clf),('rf',rf_clf),('svm',svm_clf)],voting='soft')
    
    data = data_reader.getAllData(shuffle=True)  # we shuffle the data so we can do Cross-Validation

    # TODO: add deepfake model here
    test = torch.load("module.pth",map_location=torch.device('cpu'))
    net = Network()
    net.load_state_dict(test)
    AntiSpoofClassifier = Model(net, "adam", rmse, device)
    detect_deepfakes(data,AntiSpoofClassifier)

    num_folds = 10
    #print (data[0])

    fold_length = math.floor(len(data[0]) / num_folds)
    average_accuracy = 0.0  # the performance measure of the system
    
    for clf in (kn_clf, bl_clf, rf_clf, svm_clf, voting_clf,voting_clf2):
        for k in range(num_folds):
        # get train data and test data from data
            train_data, test_data = [None, None], [None, None]
            for i in range(2):
                if k == num_folds - 1:
                    train_data[i] = data[i][:k * fold_length]
                    test_data[i] = data[i][k * fold_length:]

                else:
                    train_data[i] = data[i][:k * fold_length] + data[i][(k + 1) * fold_length:]
                    test_data[i] = data[i][k * fold_length:(k + 1) * fold_length]
            train_data, test_data = tuple(train_data), tuple(test_data)

            # compute the eigenfaces and prepare the training data to train the classifier
            X_train = feature_extractor.fit_transform(train_data[0])  # computes eigenfaces and prepares training data
            y_train = np.array(train_data[1])  # prepares training labels
        
            clf.fit(X_train, y_train)  # trains the classifier
            
            # test the performance (accurancy) of the classifier on the current fold
            X_test = feature_extractor.transform(test_data[0])  # prepares the test data
            y_test = np.array(test_data[1])  # prepares the test labels
            if clf is voting_clf2:
                matching_scores = clf.predict_proba(X_test)
                gen_scores = []
                imp_scores = []
                classes = clf.classes_
                matching_scores = pd.DataFrame(matching_scores, columns=classes)
                for i in range(len(y_test)):    
                    scores = matching_scores.loc[i]
                    mask = scores.index.isin([y_test[i]])
                    gen_scores.extend(scores[mask])
                    imp_scores.extend(scores[~mask])
                performance_plots.performance(gen_scores, imp_scores, 'VotingClassifier', 500)
            
            average_accuracy += clf.score(X_test, y_test)  # accumulates the accuracies on each fold, we'll divide it by the number of folds later
            
        average_accuracy /= num_folds  # computes the average accuracy of the classifier over all the folds
        
        print(clf.__class__.__name__,'Average accuracy: {0}%'.format(round(100 * average_accuracy), 2))
        #print(clf.__class__.__name__,'Average accuracy: {0}%'.format(round(100 * average_accuracy), 2))


    # save the classifier and the eigenfaces model for direct use via the main.py script
    # save the classifier
    f = open('clf.pkl', 'wb')
    pickle.dump(clf, f)
    f.close()

    # save the eigenfaces model
    f = open('feature_extractor.pkl', 'wb')
    pickle.dump(feature_extractor, f)
    f.close()

# def retry():
#     try:
#         main()
#         return
#     except Exception as e:
#         retry()  
# retry()

if __name__ == "__main__":
    main()