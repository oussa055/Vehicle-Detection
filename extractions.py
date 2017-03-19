import numpy as np
import cv2
import matplotlib.image as mpimg
from features import bin_spatial, color_hist,  get_hog_features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis = False):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'YCrCb'
    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))  
        else:
            if vis == True:
                    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis= True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)



    
def extract_features(imgs,spatial_size=(32, 32),hist_bins=32, orient=9,pix_per_cell=8,cell_per_block=2):    
    #1) Define an empty list to receive features
    features = []
    #2) Apply color conversion if other than 'RGB'
    for file in imgs:
        file_features = []
        img = mpimg.imread(file)
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
            
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
        
        hog_features = []
        for channel in range(3):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))  
        hog_features = np.ravel(hog_features)
        file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return features