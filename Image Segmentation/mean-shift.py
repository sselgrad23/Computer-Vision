import time
import os
import random
import math
import torch
import numpy as np
import torch
from torch import tensor
from torch import linalg as LA

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    # Computes the distance between a given point and all other points
    unsqueezed_x = torch.unsqueeze(x, dim=0)
    return(torch.cdist(unsqueezed_x, X))

def distance_batch(x, X):
    # Batchified version of distance(x,X)
    return torch.cdist(x, X)

def gaussian(dist, bandwidth):
    # Computes the weights of points according to the distance computed by the distance function
    return torch.exp(-1 * torch.div((torch.pow(dist, 2)), (2 * pow(bandwidth, 2))))
    
def update_point(weight, X):
    # Updates point position according to weights computed by the gaussian function
    return torch.div(torch.matmul(weight, X), torch.sum(weight))

def update_point_batch(weight, X):
    # Batchified version of update_point(weight,X)
    return torch.div(torch.matmul(weight, X), torch.sum(weight, dim=0).unsqueeze(1).expand(-1, 3))

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    # Batchified version of meanshift_step(X, bandwidth=2.5)
    X_ = X.clone()
    dists = distance_batch(X, X)
    weights = gaussian(dists, bandwidth)
    X_ = update_point_batch(weights, X)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
