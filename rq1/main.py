''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images

# import get_landmarks
import numpy as np
import models
from tqdm import tqdm

import performance_plots

''' Load the data and their labels '''
#image_directory = 'E:/Facial_Biometrics_Project/rq1/BlurDatabase'
image_directory = 'E:/Male'
#image_directory = 'E:/Facial_Biometrics_Project/rq1/CannyDatabase'
#image_directory = 'E:/Facial_Biometrics_Project/rq1/GrayscaleDatabase'
df = get_images.get_images(image_directory)
# shuffle
df = df.sample(frac=1)

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
# X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

''' Matching and Decision '''
# create an instance of the classifier


num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []

gen_scores = []
imp_scores = []

for i in range(0, 300):
    clf = models.FCC(df.iloc[0]["X"].shape[0])
    #clf = models.CNN()
    query_img = df.iloc[i]["X"].__array__()
    query_label = df.iloc[i]["y"]
    
    # template_imgs = np.delete(X, i, 0)
    template_imgs = df.drop(index=i,axis=0)["X"].__array__()
    # template_labels = np.delete(y, i)
    template_labels = df.drop(index=i,axis=0)["y"].__array__()
        
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_labels))
    y_hat[template_labels == query_label] = 1 
    y_hat[template_labels != query_label] = 0

    # fit
    # leave batch size at 1 for now
    clf.fit(template_imgs, y_hat)
    

    # predict
    # Gather results
    y_pred = clf.predict(query_img)
    if y_pred == 1:
        num_correct += 1
        labels_correct.append(query_label)
    else:
        num_incorrect += 1
        labels_incorrect.append(query_label)
        
    if y_hat[i] == 1 and y_pred == 1:
        gen_scores.append(probability.item())
    else:
        imp_scores.append(probability.item())

# Print results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))    
    
performance_plots.performance(gen_scores, imp_scores, 'Equal Number of Male and Female Classmates Data', 500)