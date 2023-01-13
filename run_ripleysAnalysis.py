# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:43:38 2022

@author: Magdalena Schneider, Janelia Research Campus

Script for Ripley's Analysis of multiplexed single molecule localization data
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import maskModule as mm
import dataModule as dm
import ripleysModule as rm

tstart = time.time()


def performRipleysMultiAnalysis(path, filename, fileIDs, radii, nRandomControls=100):
    
    print(f'Cell path: {path}/{filename}')
    
    #%% Load data
    nFiles = len(fileIDs)
    locData = dm.loadLocalizationData(path, filename, fileIDs)
    
    #%% Mask

    ## Load mask from file
    #cellMask = loadMask(path, "Cell_Mask.npy", pixelsize)
    #cellMask.plot()

    ## Create mask from all localization data
    cellMask = mm.createMask(locData.allData, locData.pixelsize)
    cellMask.plot()
    cellMask.save(path, f'{filename}_mask')

    #%% Perform Ripley's analysis for all data pairs
    ripleysResults = rm.initializeResultsMatrix(nFiles)
    ripleysIntegrals = np.zeros((nFiles,nFiles))
    
    for j in range(nFiles):
        for k in range(nFiles):
            print(f'Analyzing files {fileIDs[j]} with {fileIDs[k]}...')
            if j==k:
                ripleysResults[j][k] = rm.RipleysAnalysis(locData.forest[j], radii, cellMask, nRandomControls)
            else:
                ripleysResults[j][k] = rm.CrossRipleysAnalysis(locData.forest[j], locData.forest[k], radii, cellMask, nRandomControls)
            ripleysIntegrals[j,k] = ripleysResults[j][k].ripleysIntegral_data
    
    # Normalized plot
    figsize = 30
    fig, axs = plt.subplots(nFiles, nFiles, figsize=(figsize, figsize))
    for j in range(nFiles):
        for k in range(nFiles):
            ripleysResults[j][k].plot(ci=0.95, normalized=True, showControls=True,
                                      title=f"Receptor {fileIDs[j]} with {fileIDs[k]}", labelFontsize=30, axes=axs[j][k])
            
    # Unnormalized plot
    fig, axs = plt.subplots(nFiles, nFiles, figsize=(figsize, figsize))
    for j in range(nFiles):
        for k in range(nFiles):
            ripleysResults[j][k].plot(ci=0.95, normalized=False, showControls=True,
                                      title=f"Receptor {fileIDs[j]} with {fileIDs[k]}", labelFontsize=30, axes=axs[j][k])
    
    # Print and save integral matrix
    print(f'Integral matrix:\n{ripleysIntegrals}\n')
    integralfile = os.path.join(path, f'{filename}_ripleysIntegrals')
    np.save(integralfile, ripleysIntegrals)
    
    return ripleysResults, ripleysIntegrals

def getIntegralConfidenceInterval(radii):
    lim = max(radii)-min(radii)
    return [-lim,lim] 


#%% Set file paths and parameters

cellPaths = ["./data/Cell3", "./data/Cell3", "./data/Cell3"]
filenames = ['MutuDC_6h_stimuli', 'MutuDC_6h_stimuli', 'MutuDC_6h_stimuli']
fileIDs = [1,3] #[1,2,3,4,5,6] # use list(range(1,7)) for all from 1 to 6

nRandomControls = 100
rmax = 200
radii = np.concatenate((np.arange(10, 80, 2), np.arange(80, rmax, 12)))


#%% Perform Ripleys analysis over multiple receptors for each cell

allResults = []
allIntegrals = []
for path, filename in zip(cellPaths, filenames):
    ripleysResults, ripleysIntegrals = performRipleysMultiAnalysis(path, filename, fileIDs, radii=radii, nRandomControls=nRandomControls)
    allResults.append(ripleysResults)
    allIntegrals.append(ripleysIntegrals)


#%% Average Ripleys matrices over all cells
meanMatrix = np.mean( np.dstack(allIntegrals), axis=2)
print(f'Matrix of integrals over normalized Ripleys curves:\n {meanMatrix}')

#%% Confidence intervals for integrals
ci_integrals = getIntegralConfidenceInterval(radii)
print(f'Confidence interval for integral over normalized Ripleys curves:\n {ci_integrals}')


#%% Runtime
elapsedTime = time.time() - tstart
print(f'Elapsed time for whole analysis: {elapsedTime:.3f} s')
