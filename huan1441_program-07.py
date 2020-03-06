# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Author: Tao Huang (huan1441)
#
# Created: Feb 28, 2020
#
# Script: ABE65100 huan1441_program-07.py
#
# Purpose: Script to read the dataset(csv) about all Earthquakes for the past 30 days,
#          and generate six figures for the graphical analysis.
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# import the earthquake data (csv) and store into 'data'

data = pd.read_table('all_month.csv',sep=',')

# function to delete the NaN in the data

def clean(rawdata):
    output = rawdata[(np.isnan(rawdata) == False)]
    return output

# generate a histogram of earthquake magnitude
# using 10 bins with width of 1 and a range of 0 to 10

mag_data = clean(data['mag'])

plt.hist(mag_data,bins=10,width=1,range=(0,10),density=True)
plt.xlabel("Earthquake Magnitude (bins = 10)")
plt.ylabel("Density")
    
plt.savefig("Histogram of Mag_bins_10.jpeg")

plt.close()

# generate a KDE plot of earthquake magnitude
# gaussian kernel type with kernel width 0.5

mag_kde = stats.gaussian_kde(mag_data)
mag_kde.covariance_factor = lambda : 0.5
mag_kde._compute_covariance()

mag_data = np.sort(mag_data)

plt.plot(mag_data,mag_kde(mag_data))
plt.xlabel("Earthquake Magnitude")
plt.ylabel("Density")
    
plt.savefig("KDE of Earthquake Magnitude.jpeg")

plt.close()

# generate a scatter plot of latitude vs. longitude for all earthquakes

plt.plot(data['longitude'],data['latitude'],'ro')
plt.xlabel("Longitude (dgree)")
plt.ylabel("Latitude (dgree)")

plt.savefig("Latitude vs Longitude for All Earthquakes.jpeg")

plt.close()

# generate a normalized cumulative distribution plot of earthquake depths

dep_data = clean(data['depth'])

x = np.sort(dep_data)
y = np.linspace(0,1,len(dep_data))

plt.plot(x,y)
plt.xlabel("Earthquake Depth (km)")
plt.ylabel("Cumulative Density")
    
plt.savefig("CDF of Earthquake Depth.jpeg")

plt.close()

# generate a scatter plot of earthquake magnitude vs. depth

plt.plot(data['mag'],data['depth'],'bo')
plt.xlabel("Earthquake Magnitude")
plt.ylabel("Earthquake Depth (km)")
    
plt.savefig("Earthquake Magnitude vs Depth.jpeg")

plt.close()

# generate a Q-Q plot of the earthquake magnitudes

stats.probplot(mag_data, dist="norm", plot=plt)
plt.ylabel("Earthquake Magnitude")

plt.savefig("Q-Q plot of Earthquake Magnitudes.jpeg")

plt.close()
