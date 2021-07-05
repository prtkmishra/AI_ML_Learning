"""
********************************************************************************************
file: image_processing.py
author: @Prateek Mishra
Description: Image Processing
********************************************************************************************
"""
"""Import Modules"""
import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color 
import skimage.feature as feature
import skimage.util as util
import skimage.segmentation as segmentation
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage

"""Define input images"""
avengers = "./data/avengers_imdb.jpg"
forestry = "./data/forestry_commission_gov_uk.jpg"
rolland = "./data/rolland_garros_tv5monde.jpg"
bushHouse = "./data/bush_house_wikipedia.jpg"

"""Read image files"""
AV = imageio.imread(avengers)
BH = imageio.imread(bushHouse)
FR = imageio.imread(forestry)
RL = imageio.imread(rolland)

"""Define Functions for each question"""
"""@question2_1
        Input: image
        Method: Convert to Grayscale
        define threshold using OTSU method
        identify shape of the image
        plot grayscale image and binary image"""
def question2_1(image):
    grey = color.rgb2gray(image)
    thresh = threshold_otsu(grey,nbins=256)
    bin = grey > thresh
    x,y = grey.shape
    print("Shape of avengers_imdb : ",(y,x),"and total size:",image.size)
    fig, (ax0, ax1, ax2) = plt.subplots( nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True )

    ax0.imshow(image)
    ax0.axis( 'off' )
    ax0.set_title( 'Original image' )

    ax1.imshow( grey, cmap=plt.cm.gray)
    ax1.axis( 'off' )
    ax1.set_title( 'Greyscale image' )

    ax2.imshow( bin, cmap=plt.cm.gray)
    ax2.axis( 'off' )
    ax2.set_title( 'Binary image' )
    fig.tight_layout()
    plt.savefig( './output/question2_1.png' )
    # plt.show()

"""
@question2_2
Input: image
Method: apply gaussian noise to image
        apply gaussian filter
        apply uniform filter
output:
        plot images
"""
def question2_2(image):
    noise = util.random_noise(image, mode='gaussian', seed=None, clip=True, var=0.1)
    gaussianFilter = gaussian(noise, sigma=1, multichannel=True)
    uniformSmoothing = ndimage.uniform_filter(gaussianFilter, size=(9,9,1))
    fig, (ax0, ax1,ax2, ax3) = plt.subplots( nrows=1, ncols=4, figsize=(12, 4), sharex=True, sharey=True )

    ax0.imshow(image)
    ax0.axis( 'off' )
    ax0.set_title( 'Original image' )

    ax1.imshow(noise)
    ax1.axis( 'off' )
    ax1.set_title( 'Noisy image' )
    
    ax2.imshow(gaussianFilter)
    ax2.axis( 'off' )
    ax2.set_title( 'Gaussian Smoothing' )

    ax3.imshow(uniformSmoothing)
    ax3.axis( 'off' )
    ax3.set_title( 'Uniform Smoothing' )

    fig.tight_layout()
    plt.savefig( './output/question2_2.png' )
    # plt.show()
"""
@question2_3
Input: image
Method: apply gaussian for smoothing
        apply kmeans segmentation method using segmentation.slic
output:
        plot images
"""        
def question2_3(image):
    gaussianFilter = gaussian(image, sigma=3, multichannel=True)
    # seg = segmentation.slic(gaussianFilter, n_segments=5, max_iter=50)
    seg = segmentation.slic(gaussianFilter, n_segments=5, max_iter=10, enforce_connectivity=True)
    fig, (ax0, ax1) = plt.subplots( nrows=1, ncols=2, figsize=(20, 10), sharex=True, sharey=True )

    ax0.imshow(image)
    ax0.axis( 'off' )
    ax0.set_title( 'Original image' )

    ax1.imshow(seg)
    ax1.axis( 'off' )
    ax1.set_title( 'Segmented image' )
    fig.tight_layout()
    plt.savefig( './output/question2_3.png' )
    # plt.show()
"""
@question2_4
Input: image
Method: Convert to gray scale
        apply canny edge detector withough sigma
        apply canny edge detector with sigma
        Apply logical operation on both canny edge outputs
        Feed logical output to hough transformation function hough_line
output:
        plot images
"""        
def question2_4(image):
    grey = color.rgb2gray(image)
    ed1 = feature.canny(grey,low_threshold=0.1,high_threshold=0.9,use_quantiles=True)
    ed2 = feature.canny(grey,sigma=2.5,low_threshold=0.1,high_threshold=0.9,use_quantiles=True)
    ed = ed1^ed2
    h, theta, d = hough_line(ed)

    fig, ((ax0, ax2),(ax1,ax3)) = plt.subplots( nrows=2, ncols=2, figsize=(12, 4), sharex=True, sharey=True )

    ax0.imshow(image)
    ax0.axis( 'off' )
    ax0.set_title( 'Original image' )

    ax1.imshow(ed, cmap='gray')
    ax1.axis( 'off' )
    ax1.set_title( 'Canny image' )

    ax2.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
           d[-1], d[0]], cmap='gray', aspect=1/1.5)
    ax2.set_title('Hough transform')
    ax2.set_xlabel('Angles (degrees)')
    ax2.set_ylabel('Distance (pixels)')
    ax2.axis('image')

    ax3.imshow(image)
    row1, col1 = grey.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
        ax3.plot((0, col1), (y0, y1), '-b')
    ax3.axis((0, col1, row1, 0))
    ax3.set_title('Detected lines')
    ax3.set_axis_off()


    fig.tight_layout()
    plt.savefig('./output/question2_4.png')
    # plt.show()

question2_1(AV)
question2_2(BH)
question2_3(FR)
question2_4(RL)
