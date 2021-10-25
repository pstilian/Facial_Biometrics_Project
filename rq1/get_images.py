# Load imports
import os 
import cv2
import pandas as pd

'''
function get_images() definition
Parameter, image_directory, is the directory 
holding the images
'''

def get_images(image_directory):
    # X = []
    # y = []
    df = pd.DataFrame(columns=["X","y"])
    extensions = ('jpg','png','gif')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    # read the image using openCV                    
                    img = cv2.imread(
                            os.path.join(image_directory, subfolder, file)
                            )
                    # resize the image
                    # they must be 224 * 224 since this is what ViT expects
                    # however, for FCC I am using 64 * 64 to prevent the parameters from blowing up                     
                    width = 16
                    height = 16
                    dim = (width, height)
                    img = cv2.resize(img, dim)
                    # add the resized image to a list X
                    # X.append(img)
                    # add the image's label to a list y
                    # y.append(subfolder)
                    df = df.append({"X":img,"y":subfolder},ignore_index=True)
    
    print("All images are loaded")     
    # return the images and their labels      
    # return X, y
    return df
                    
                
            