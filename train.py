#%%
'''
This python file contains the code to detect the spices in the spic box. 
For the sake of poc, the 2 spices, cloves and cinnamon stick have been trained into the model for the detection.
The Convolutional Neural Network algorithm is used for the spice detection.
Link to reference: https://pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
'''

#%%
'''Importing the libraries'''
import matplotlib
matplotlib.use('Agg')

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import adam_v2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import cv2
import os

