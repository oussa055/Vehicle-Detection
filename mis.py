import matplotlib.pyplot as plt
import numpy as np

def visualize(fig, rows, columns, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, columns, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap = 'hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])