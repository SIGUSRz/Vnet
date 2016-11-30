import numpy as np
import tensorflow as tf
import scipy

def load_image(image_file):
    img = scipy.misc.imread(image_file)
    # GrayScale
    if len(img.shape) == 2:
        img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='float32')
        img_new[:,:,0] = img
        img_new[:,:,1] = img
        img_new[:,:,2] = img
        img = img_new
    resized_img = scipy.misc.imresize(img, (224, 224))
    return (resized_img / 225.0).astype('float32')