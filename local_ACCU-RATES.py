# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023
from analyse_SAC_CAM30.py
@author: pmartins
"""

# Package imports
# import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import io
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import t as tt
from scipy.special import lambertw
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import csv


# # Simulation
# t = np.linspace(0,25,100)
# S0 = [0.1, 0.5, 1, 5, 10, 50]
# noise = 0.01
# Kms = 5
# Vms = 1
# Ncurves = len(S0)
# # colist = ['Curve' for i in range(1, 2*Ncurves+1)]
# data = {}
# data2 = {}
# data2.update({'Time (s)': t})
# for i in range(Ncurves):
#     lambert = lambertw(S0[i]/Kms*np.exp((S0[i]-Vms*t)/Kms))
#     Pt = S0[i] - Kms*lambert.real #S0[i] -
#     data.update({str(2*i): t})
#     data.update({str(2*i+1): Pt})
#     data2.update({str(S0[i]) + ' uM': Pt})            
# df = pd.DataFrame(data)
# df2 = pd.DataFrame(data2)
# datast = 'simulation_pt100'
# filepath = Path('subfolder/in_' + datast + ".csv")
# with open(filepath, 'w') as the_file:
#     the_file.write('*** Simulated Data ***\n')
# df2.to_csv(filepath, mode='a', index=False)

def load_odf():
    """ Get the loaded .odf into Pandas dataframe. """
    try:
        df_load = pd.read_excel(input, skiprows=3,  engine="odf", header = None)
        S0_load = df_load.iloc[0]
        S0_load.dropna(how='all', axis=0, inplace=True)
        S0 = list(S0_load)
        df_load = df_load.iloc[4:]
        # Remove empty columns
        df_load.dropna(how='all', axis=1, inplace=True)
        # number of curves
        Ncurves = S0_load.shape[0]
        # Curve1, Curve2...
        colist = ['Curve '+ str(i) for i in range(1, Ncurves+2)]
        # df_load.columns =  np.concatenate((['Time'], colist), axis=0)
        flag_return = 0
    except:
        df_load = 0
        Ncurves  = 0
        colist = 0
        flag_return = 1
    return df_load, S0, Ncurves, colist, flag_return
datast = 'gmm_control_tfin3_5e4' #ACCU-RATES_template Sdepletion gmm_control
input = "datasets//" + datast + ".ods"
df, S0, Ncurves, colist, flag_return = load_odf()

colormap = plt.cm.coolwarm 
colors = [colormap(i) for i in np.linspace(0, 1, Ncurves)]

# Output *.csv file
filepath = Path("subfolder/out_"  + datast + "_New.csv")    
filepath.parent.mkdir(parents=True, exist_ok=True)  
with open(filepath, 'w') as the_file:
    the_file.write('*** ACCU-RATES REPORT ***\n')

# Check if at least 5 different S0 values are present 
if len(S0) < 5:
    with open(filepath, 'a') as the_file:
        the_file.write('\nACCU-RATES method not applied. Check if at least 5 different S0 values are present.')
    # analysisterminated()

def extract_curves(dfi):
    # remove empty cells
    dfi = dfi.dropna()
    ydata = dfi.iloc[:, 1]
    xdata = dfi.iloc[:, 0]
    return xdata, ydata
    
# DATA ANALYSIS AND REPORT

# If S vs t curves are used, conversion to P vs t is needed        
mint = []
maxt = []
minP = []
maxP = []
SvstFlag = 0
for n in range(Ncurves):
    dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
    t, P = extract_curves(dfi)
    # mint.extend([min(t)])
    # maxt.extend([max(t)])
    # minP.extend([min(P)])
    # maxP.extend([max(P)])
    if list(P)[-1] < list(P)[0]:
        SvstFlag = SvstFlag + 1
if SvstFlag == Ncurves:
    with open(filepath, 'a') as the_file:
        the_file.write('\nSubstrate depletion curves were detected. Conversion to' + 
                       ' Product production curves was performed using the approximation P = S0-S.\n')
# mint = min(mint)
# maxt = max(maxt)
# minP = min(minP)
# maxP = max(maxP)
    
# (1) Reaction Timepoint Number & Moving Average Filter------------------------

# Defining number of final sampled points
nosample = 7

pointnumber = []
for n in range(Ncurves):
    dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
    xdata, ydata = extract_curves(dfi)
    pointnumber.extend([len(xdata)]);
    
# Fit full progress curve
def integratedMM(x, param0, param1, S):
    K = param0
    V = param1
    # S = param2
    y = S - K*(lambertw(S/K*np.exp((S-V*x))/K))
    return y

fig3, ax3 = plt.subplots()
Km_int = []
Vmax_int = []
S0_int = []
for n in range(Ncurves):
    try:
        dfi = df.iloc[:, [2*n, 2*n+1]].astype(float)
        t, P = extract_curves(dfi)
        if SvstFlag == Ncurves:
            P = S0[n] - P
        ig = np.asarray([max(S0)/5, max(S0)/max(t)])
        bds =  ([0, 0], [np.inf, np.inf])
        parameters, covariance = curve_fit(lambda x, a, b: integratedMM(x, a, b, S0[n]), t, P, p0=ig,
                                        bounds=bds, maxfev=10000) #xtol=1e-20*tmax, ftol=1e-20*ymax, 
        
        Km_int.extend([parameters[0]])
        Vmax_int.extend([parameters[1]])
        # S0_int.extend([parameters[2]])
        plt.scatter(t, P, color=colors[n], label=str(S0[n]))
        p_xfit = np.linspace(0, max(t))
        plt.plot(p_xfit, integratedMM(p_xfit, parameters[0], parameters[1], S0[n]),
                  'b', label='_fitted line')
        plt.legend(title="S0")
    except:
        # S0_int.extend([S0[n]])
        Km_int.extend([0])
        Vmax_int.extend([0])
        


# Flag is 1 is at least one vector is found to have less than 3 points.    
pointnumber_flag = all(i < 3 for i in pointnumber)
if not(pointnumber_flag):
    
    percentile95 = []
    index95 = []
    intfilt_all = []
    for n in range(Ncurves):
        individualfitting = 1
        dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
        t, P = extract_curves(dfi)
        if SvstFlag == Ncurves:
            P = S0[n] - P
        percentile95.extend([np.percentile(P, 95)])
        index95.extend([(np.abs(P - percentile95[n])).argmin()])
        intfilt_all.extend([round(index95[n]/nosample)]);
    # Definition of largest interval for strong filter application
    intfilthalf = round(max(intfilt_all)/2) if round(max(intfilt_all)/2) >0 else 1 ;
    
    Pfiltered = []
    tfiltered = []
    for n in range(Ncurves):
        individualfitting = 1
        dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
        t, P = extract_curves(dfi)
        if SvstFlag == Ncurves:
            P = S0[n] - P
        filtereddata = []
        filteredtime = []
        for q in range(intfilthalf, (pointnumber[n]-intfilthalf+2)):
            filtereddata.extend([np.mean( P[(q - intfilthalf):(q + intfilthalf)] )])  
            filteredtime.extend([np.mean( t[(q - intfilthalf):(q + intfilthalf)] )])
        Pfiltered.append(np.transpose(filtereddata))
        tfiltered.append(np.transpose(filteredtime))
    
   
    # Sampling x Points, Product-wise
    Psampled = []
    tsampled = []
    UniqueFlag = 0
    for n in range(Ncurves):
        tempP = Pfiltered[n]
        tempt = tfiltered[n]
        productsamples = np.linspace(min(tempP) , max(tempP), nosample )
        sampleP = []
        samplet = []
        for q in range(nosample):
            closestindex = (np.abs(tempP - productsamples[q])).argmin()
            sampleP.extend([tempP[closestindex]])
            samplet.extend([tempt[closestindex]])
        # check if arrays are unique
        if len(sampleP) > len(set(sampleP)):
            UniqueFlag = 1
        Psampled.append(np.transpose(sampleP))
        tsampled.append(np.transpose(samplet))
        
# Less than 3 timepoints/curve or non-unique  
else:
    individualfitting = 0
    Psampled = []
    tsampled = []
    for n in range(Ncurves):
        tempP = Pfiltered[n]
        tempt = tfiltered[n]
        Psampled.append(np.transpose(tempP))
        tsampled.append(np.transpose(tempt))
    with open(filepath, 'a') as the_file:
        the_file.write('\nACCU-RATES method not applied. Number of timepoints < 3.\n')

# Non-unique arrays
if UniqueFlag == 1:
    individualfitting = 0
    Psampled = []
    tsampled = []
    for n in range(Ncurves):
        tempP = Pfiltered[n]
        tempt = tfiltered[n]
        Psampled.append(np.transpose(tempP))
        tsampled.append(np.transpose(tempt))
       
    
# (2) Individual Curve Analysis------------------------------------------------
# Choosing whether to use Pinf or not.
eval = []
FinalP = []
for n in range(Ncurves):
    eval.extend([(Psampled[n][-1] > 0.9*S0[n]) & (Psampled[n][-1] < 1.1*S0[n])])
for n in range(Ncurves):
    # if all(eval): # & SvstFlag != Ncurves
    #     FinalP.extend([S0_int[n]]) #max(Psampled[n])
    # else: 
    #     FinalP.extend([S0[n]])
     FinalP.extend([S0[n]])
        
# Determination of Y and X for ti = first points. 
# Unlike Interferenzy, a single initial timpoint (the second) is assumed 

    # linearlization
X = []
Y = []
for n in range(Ncurves):
    linearX = -np.log(1 - (Psampled[n][2] - Psampled[n][1]) / 
                      (FinalP[n] - Psampled[n][1])) / (tsampled[n][2] - tsampled[n][1])
    linearY = (Psampled[n][2] - Psampled[n][1]) / (tsampled[n][2] - tsampled[n][1])
    X.extend([linearX])
    Y.append(np.transpose(linearY))

res = stats.linregress(np.multiply(X,-1),Y)
# Two-sided inverse Students t-distribution
# p - probability, dfr - degrees of freedom
tinv = lambda p, dfr: abs(tt.ppf(p/2, dfr))
ts = tinv(0.05, len(X)-2)
with open(filepath, 'a') as the_file:
    the_file.write('\nKinetic Parameters:')
with open(filepath, 'a') as the_file:
    the_file.write('\nKm (95%): ' + repr(round(res.slope,6)) + 
                   ' +/- ' + repr(round(res.stderr,6)))
with open(filepath, 'a') as the_file:
    the_file.write('\nVmax (95%): ' + repr(round(res.intercept,6)) + 
                   ' +/- ' + repr(round(res.intercept_stderr,6)))
Km = res.slope
Vmax = res.intercept

with open(filepath, 'a') as the_file:
    the_file.write('\n\nInitial Rates:\n')
v0 = []
for n in range(Ncurves):
    v0.extend([S0[n]*Vmax / (S0[n] + Km)])
v0_df = pd.DataFrame({'S0': S0,
           'v0': v0})
v0_df.to_csv(filepath, mode='a', index=False)

p_xfit = np.linspace(0, max(X), 200)
fig1, ax1 = plt.subplots()
plt.plot(X, Y, 'o', label='linear coordinates')
plt.plot(p_xfit, res.intercept - res.slope*p_xfit, 'r', label='fitted line')
plt.legend()

# Plot linearization fit
fig2, ax2 = plt.subplots()
Pmax = max([max(Psampled[i]) for i in range(Ncurves)])
p_tmax = Pmax/max(v0)
p_xfit = np.linspace(0, p_tmax, 200)
for n in range(Ncurves):
    dfi = df.iloc[:, [2*n,2*n+1]].astype(float)
    t, P = extract_curves(dfi)
    if SvstFlag == Ncurves:
        P = S0[n] - P
    plt.scatter(t, P, color=colors[n], label=str(S0[n]))
    plt.plot(p_xfit, v0[n]*(p_xfit - tsampled[n][1]) + Psampled[n][1], 'r', label='_fitted line')
plt.legend(title="S0")


#MM curve
fig4, ax4 = plt.subplots()
plt.scatter(S0, v0, color = 'r', s = 80, label='ACCU-RATES')
p_xfit = np.linspace(0, max(S0))
plt.plot(p_xfit, Vmax*p_xfit / (Km + p_xfit), 'r', label='_fitted line')

v0_int = []
for n in range(Ncurves):
    labelMM = 'Approximate' if n==0 else '_Approximate'
    v0_int.extend([Vmax_int[n]*S0[n] / (Km_int[n] + S0[n])]) 
    plt.scatter(S0[n], v0_int[n], color = 'b', 
                s=50, label=labelMM)
plt.legend()

with open(filepath, 'a') as the_file:
    the_file.write('\n***********************\n' + '*Approximate* Kinetic Parameters:\n')

# Fit MM curve
def MM(x, param0, param1):
    K = param0
    V = param1
    y = x*V/(K+x)
    return y

ig = np.asarray([max(S0)/5, max(v0)])
bds =  ([0, 0], [np.inf, np.inf])
parameters, covariance = curve_fit(MM, S0, v0_int, p0=ig,
                               bounds=bds, maxfev=10000) #xtol=1e-20*tmax, ftol=1e-20*ymax, 

with open(filepath, 'a') as the_file:
    the_file.write('Km: ' + repr(round(parameters[0],6)))
with open(filepath, 'a') as the_file:
    the_file.write('\nVmax: ' + repr(round(parameters[1],6)))
     
with open(filepath, 'a') as the_file:
    the_file.write('\n\n*Approximate* Initial Rates:\n')
v0int_df = pd.DataFrame({'S0': S0,
           'v0': v0_int})
v0int_df.to_csv(filepath, mode='a', index=False)