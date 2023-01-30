# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:41:53 2022

@author: Magdalena Schneider, Janelia Research Campus
"""

import unittest
import numpy as np
import dataModule as dm
import maskModule as mm
import ripleysModule as rm

np.random.seed(10) # initialize random seed

class TestRandomDistribution(unittest.TestCase):
    
    def test_randomCase(self):
        path = "./data/Cell3"
        filename = 'MutuDC_6h_stimuli'
        pixelsize = 130
        cellMask = mm.loadMask(path, f'{filename}_mask.npy', pixelsize)
        
        nPoints = 10000
        points, *__ = cellMask.randomPoints(nPoints)
        radii = np.arange(10, 50, 2)
        cellMask.plotPoints(points, title='Test data')
        
        results = rm.RipleysAnalysis(points, radii, cellMask, nControls=200)
        results.plot(ci=0.95, normalized=True, showControls=True, title='Ripleys (normalized)')
        results.plot(ci=0.95, normalized=False, showControls=True, title='Ripleys')
        
    def test_cross_randomCase(self):
        path = "./data/Cell3"
        filename = 'MutuDC_6h_stimuli'
        pixelsize = 130
        cellMask = mm.loadMask(path, f'{filename}_mask.npy', pixelsize)
        
        nPoints = 10000
        pointsA, *__ = cellMask.randomPoints(nPoints)
        pointsB, *__ = cellMask.randomPoints(nPoints)
        radii = np.arange(10, 50, 2)
        cellMask.plotPoints(pointsA, title='Test data 1')
        cellMask.plotPoints(pointsB, title='Test data 2')
        
        results = rm.CrossRipleysAnalysis(pointsA, pointsB, radii, cellMask, nControls=200)
        results.plot(ci=0.95, normalized=True, showControls=True, title='Cross Ripleys (normalized)')
        results.plot(ci=0.95, normalized=False, showControls=True, title='Cross Ripleys')
        
    def test_cross_noInteraction_clusteredA_randomB(self):
        path = "./data/Cell3"
        filename = 'MutuDC_6h_stimuli'
        pixelsize = 130
        cellMask = mm.loadMask(path, f'{filename}_mask.npy', pixelsize)
        
        nPoints = 10000
        pointsA, *__ = cellMask.randomPoints(nPoints)
        ind = int(nPoints/3)
        pointsA = np.vstack([pointsA[0:ind,:], pointsA[0:ind,:], pointsA[0:ind,:]])
        
        pointsB, *__ = cellMask.randomPoints(nPoints)
        radii = np.arange(10, 50, 2)
        cellMask.plotPoints(pointsA, title='Test data 1')
        cellMask.plotPoints(pointsB, title='Test data 2')
        
        results = rm.CrossRipleysAnalysis(pointsA, pointsB, radii, cellMask, nControls=200)
        results.plot(ci=0.95, normalized=True, showControls=True, title='Cross Ripleys (normalized)')
        results.plot(ci=0.95, normalized=False, showControls=True, title='Cross Ripleys')
        
    def test_cross_noInteraction_randomA_clusteredB(self):
        path = "./data/Cell3"
        filename = 'MutuDC_6h_stimuli'
        pixelsize = 130
        cellMask = mm.loadMask(path, f'{filename}_mask.npy', pixelsize)
        
        nPoints = 10000
        pointsA, *__ = cellMask.randomPoints(nPoints)
        pointsB, *__ = cellMask.randomPoints(nPoints)
        ind = int(nPoints/3)
        pointsB = np.vstack([pointsB[0:ind,:], pointsB[0:ind,:], pointsB[0:ind,:]])
        radii = np.arange(10, 50, 2)
        cellMask.plotPoints(pointsA, title='Test data 1')
        cellMask.plotPoints(pointsB, title='Test data 2')
        
        results = rm.CrossRipleysAnalysis(pointsA, pointsB, radii, cellMask, nControls=200)
        results.plot(ci=0.95, normalized=True, showControls=True, title='Cross Ripleys (normalized)')
        results.plot(ci=0.95, normalized=False, showControls=True, title='Cross Ripleys')

if __name__ == '__main__':
    unittest.main()