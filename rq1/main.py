''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from torch.optim import Adam
# import get_landmarks
import numpy as np
from tqdm import tqdm

''' Load the data and their labels '''
image_directory = 'Caltech Faces Dataset'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
# X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

''' Matching and Decision '''
# create an instance of the classifier
clf = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to("cuda")
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
optimizer = Adam(clf.parameters(),lr=1e-5)

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
    
    template_imgs = np.reshape(template_imgs,(412,3,224,224))
    # template_imgs = feature_extractor(images=template_imgs, return_tensors="pt")

    # fit
    # leave batch size at 1 for now
    clf.train()
    for img in tqdm(range(template_imgs.shape[0])):
        label = torch.tensor(y_hat[img],dtype=torch.long).to('cuda')
        img = feature_extractor(images=template_imgs[img], return_tensors="pt")
        img = img.to('cuda')
        outputs = clf(**img,labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    

    # predict
    # Gather results
    clf.eval()
    with torch.no_grad():
        img = feature_extractor(images=query_img, return_tensors="pt")
        img = img.to('cuda')
        outputs = clf(**img)
        logits = outputs.logits
        y_pred = logits.argmax(-1).item()
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
    
    