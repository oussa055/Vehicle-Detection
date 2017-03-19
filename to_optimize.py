def findcars(img, scale, pix_per_cell = 8, orient = 9, cell_per_block = 2, spatial_size = spatial_size_val,
             hist_bins = hist_bins_val, y_range = (None, None)):
    draw_img = np.copy(img)
    img_boxes = []
    plt.imshow(draw_img)
    plt.show()
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255
    img_to_search = img[y_range[0]:y_range[1],:,:]
    ctrans_tosearch = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    #nnumber of blocks 
    nxblocks = (ch1.shape[1]//pix_per_cell) - 1
    nyblocks = (ch2.shape[0]//pix_per_cell) - 1
        
    nfeat_per_block = orient * cell_per_block ** 2
    windows = 64
        
    nblocks_per_window = (windows//pix_per_cell) - 1
    cell_per_step = 2
        
    nxsteps = (nxblocks - nblocks_per_window)//cell_per_step
    nysteps = (nyblocks - nblocks_per_window)//cell_per_step
        
    #HOG for whole image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec = False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec = False)
    hog3, hog3_img = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec = False, vis = True)
    print (hog2.shape)
    plt.imshow(hog3_img)
    plt.show()
        
    for xb in range(nxsteps):
        for yb in range (nysteps):

            ypos = yb * cell_per_step
            xpos = xb * cell_per_step
                
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
                
            subimg = cv2.resize(ctrans_tosearch[ytop: ytop+windows, xleft:xleft+windows], (64,64))
                
            #color and spatial binning 
            spatial_feature = bin_spatial(subimg, size=(spatial_size_val))
                
            hist_features = color_hist(subimg, nbins = hist_bins_val)
            features = np.hstack((spatial_feature, hist_features, hog_features)).reshape(1,-1)
            test_features = X_scalar.transform(features)
                
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(windows*scale)
                print (ytop_draw, y_range[0])    
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+y_range[0]),
                                  (xbox_left+windows, ytop_draw + windows+ y_range[0]), (0,0,255), 6)
                img_boxes.append(((xbox_left, ytop_draw+y_range[0]),
                                  (xbox_left+windows, ytop_draw + windows+ y_range[0])))
                heatmap[ytop_draw+y_range[0]:ytop_draw+windows+y_range[0], xbox_left: xbox_left+windows] += 1
    plt.imshow(draw_img)
    plt.show()