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

np.random.seed(0) # initialize random seed

tstart = time.time()


def performRipleysMultiAnalysis(path, filename, fileIDs, radii, nRandomControls=100):
    
    print(f'Cell path: {path}/{filename}')
    
    #%% Load data
    nFiles = len(fileIDs)
    locData = dm.loadLocalizationData(path, filename, fileIDs)

    #%% Create subfolder for results
    results_path = os.path.join(path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #%% Mask

    ## Load mask from file
    #cellMask = loadMask(path, "Cell_Mask.npy", pixelsize)
    #cellMask.plot()

    ## Create mask from all localization data
    cellMask = mm.createMask(locData.allData, locData.pixelsize)
    cellMask.plot()
    cellMask.save(results_path, f'{filename}_mask')


    #%% Perform Ripley's analysis for all data pairs
    ripleysResults = rm.initializeResultsMatrix(nFiles)
    ripleysIntegrals = np.zeros((nFiles,nFiles))
    
    for j in range(nFiles):
        for k in range(nFiles):
            print(f'Analyzing interaction between receptor {fileIDs[j]} and {fileIDs[k]}...')
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
    fig.savefig(os.path.join(results_path, f'{filename}_ripleys_normalized'))
    
            
    # Unnormalized plot
    fig, axs = plt.subplots(nFiles, nFiles, figsize=(figsize, figsize))
    for j in range(nFiles):
        for k in range(nFiles):
            ripleysResults[j][k].plot(ci=0.95, normalized=False, showControls=True,
                                      title=f"Receptor {fileIDs[j]} with {fileIDs[k]}", labelFontsize=30, axes=axs[j][k])
    fig.savefig(os.path.join(results_path, f'{filename}_ripleys_unnormalized'))
    
    # Print and save integral matrix
    print(f'Integral matrix:\n{ripleysIntegrals}')
    integralfile = os.path.join(results_path, f'{filename}_ripleysIntegrals')
    np.save(integralfile, ripleysIntegrals)
    np.savetxt(integralfile+'.dat', ripleysIntegrals, delimiter='\t')

    print(f'Results saved in {results_path}\n')
    
    return ripleysResults, ripleysIntegrals

def getIntegralConfidenceInterval(radii):
    lim = float(max(radii)-min(radii))
    return [-lim,lim] 


#%% Set file paths and parameters

# NOTE: Change paths and filenames to the actual data, same file is taken multiple times here for demonstration purpose only
cellPaths = ["./data/Cell3"]
filenames = ['MutuDC_6h_stimuli']
fileIDs = list(range(1,7))

nRandomControls = 100
rmax = 200
radii = np.concatenate((np.arange(4, 80, 2), np.arange(80, rmax+1, 12)))


#%% Perform Ripleys analysis over multiple receptors for each cell

allResults = []
allIntegrals = []
for path, filename in zip(cellPaths, filenames):
    ripleysResults, ripleysIntegrals = performRipleysMultiAnalysis(path, filename, fileIDs, radii=radii, nRandomControls=nRandomControls)
    allResults.append(ripleysResults)
    allIntegrals.append(ripleysIntegrals)


#%% Average Ripleys matrices over all cells
meanMatrix = np.mean( np.dstack(allIntegrals), axis=2)
print(f'Average integral matrix of normalized Ripleys curves over all analyzed files:\n {meanMatrix}')

#%% Confidence intervals for integrals
ci_integrals = getIntegralConfidenceInterval(radii)
print(f'Confidence interval for integral over normalized Ripleys curves:\n {ci_integrals}')


#%% Runtime
elapsedTime = time.time() - tstart
print(f'Elapsed time for whole analysis: {elapsedTime:.3f} s')
