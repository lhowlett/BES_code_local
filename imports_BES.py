import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, interp2d
from netCDF4 import Dataset
import scipy.signal as sig
import matplotlib.pyplot as plt
from astropy.convolution import convolve
from matplotlib import colors
import shapely.geometry as geom
import imageio
import math
import copy
from matplotlib.gridspec import GridSpec
from cycler import cycler

import sys
sys.path.append('/home/lh1590/Documents/PhD_Project/CodingAndData/DataFiles')
sys.path.append('/home/lh1590/Documents/PhD_Project/CodingAndData/PythonScripts/Databases')

# nicer plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

nrows = 4
ncols = 8

f_samp = 2000001.2740736576
nchan = 32
columns = [[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], 
           [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]
rows = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], 
        [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]

# smoothing function
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# butterworth low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = sig.filtfilt(b, a, data, method='gust')
    return y

# butterworth high-pass filter
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = sig.butter(order, normal_cutoff, btype='highpass', analog=False)
    y = sig.filtfilt(b, a, data, method='gust')
    return y

# butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='bandpass')
    y = sig.filtfilt(b, a, data)
    return y

# butterworth bandstop filter
def butter_bandstop_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = sig.butter(order, [low, high], btype='bandstop')
    y = sig.filtfilt(b, a, data)
    return y

# gaussian function
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# straight line in a log plot
def log_line(x, m, c):
    return (x**m)*np.exp(c)

# two gaussians
def two_gaussian(x, *pars):
    return ((x**pars[0])*np.exp(pars[1]) 
            + pars[2]*np.exp(-(x - pars[3])**2/(2*pars[4]**2)) 
            + pars[5]*np.exp(-(x - pars[6])**2/(2*pars[7]**2)))

def three_gaussian(x, *pars):
    offset = log_line(x, pars[0], pars[1])
    g1 = gaussian(x, pars[2], pars[3], pars[4])
    g2 = gaussian(x, pars[5], pars[6], pars[7])
    g3 = gaussian(x, pars[8], pars[9], pars[10])
    return g1 + g2 + g3 + offset

def four_gaussian(x, *pars):
    offset = log_line(x, pars[0], pars[1])
    g1 = gaussian(x, pars[2], pars[3], pars[4])
    g2 = gaussian(x, pars[5], pars[6], pars[7])
    g3 = gaussian(x, pars[8], pars[9], pars[10])
    g4 = gaussian(x, pars[11], pars[12], pars[13])
    return g1 + g2 + g3 + g4 + offset

def logfunc1(x, *pars):
    offset = pars[0]*x + pars[1]
    g1 = gaussian(x, pars[2], pars[3], pars[4])
    g2 = gaussian(x, pars[5], pars[6], pars[7])
    return offset + g1 + g2