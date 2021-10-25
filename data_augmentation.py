# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:26:35 2021

@author: Sreten Dedic
"""

import image_slicer
import os

image_directory = 'Input\SoF Dataset'
output_directory = 'Output\SoF Dataset'
num_slices = 36
extensions = ('jpg', 'png', 'gif')

'''
Each subject has their own folder with their
images. The following line lists the names
of the subfolders within image_directory.
'''
subfolders = os.listdir(image_directory)
for subfolder in subfolders:
    print("Creating Output directory for %s" % subfolder)
    os.mkdir(os.path.join(output_directory, subfolder))
    print("Splicing images in %s" % subfolder)
    if os.path.isdir(os.path.join(image_directory, subfolder)):  # only load directories
        subfolder_files = os.listdir(os.path.join(image_directory, subfolder))
        for file in subfolder_files:
            if file.endswith(extensions):  # grab images only
                filename, file_extension = os.path.splitext(os.path.join(image_directory, subfolder, file))
                if file_extension == '.jpg':
                    file_extension = '.jpeg'
                tiles = image_slicer.slice(os.path.join(image_directory, subfolder, file), num_slices, save=False)
                image_slicer.save_tiles(tiles, directory=os.path.join(output_directory, subfolder),
                                        prefix=file, format=file_extension.replace(".", ""))
                print(os.path.join(image_directory, subfolder, file))



print("Finished expanding dataset into " + str(num_slices) + " slices")



