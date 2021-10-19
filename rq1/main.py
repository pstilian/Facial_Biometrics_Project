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

''' Load the data and their labels '''
image_directory = 'Z:/Code/Facial_Biometrics_Project/rq1/Caltech Faces Dataset x36'
X, y = get_images.get_images(image_directory)
clf = models.CNN()

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
# X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

''' Matching and Decision '''
# create an instance of the classifier


num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []

for i in range(0, len(y)):
    query_img = X[i]
    query_label = y[i]
    
    template_imgs = np.delete(X, i, 0)
    template_labels = np.delete(y, i)
        
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

# Print results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))    
    
    