#This file will profile various functions in sparse audio
import pstats, cProfile
import numpy as np

from LCApython import lca as lcap
from LCAfortran import lca as lcaf
from LCAcython import lca as lcac
from LCAcythonv import lca as lcav

def main():
    """Profiles functions related to the network."""
    
    #Setup variables for inference
    numDict = int(800)
    numBatch = int(100)
    dataSize = int(200)
    dictsIn = np.random.uniform(size=(numDict,dataSize))
    coeffs = np.random.uniform(size=(numBatch,numDict))
    stimuli = np.random.uniform(size=(numBatch,dataSize))
    batchCoeffs = np.random.uniform(size=(numBatch,numDict))
    eta = .01
    lamb = .05
    nIter = 300
    softThresh = int(0)
    adapt = .99
    thresh = np.random.uniform(size=numBatch)
    
    #LCA
    params = """Parameters:
             numDict: """+str(numDict)+"""
             numBatch: """+str(numBatch)+"""
             dataSize: """+str(dataSize)+"""
             nIter: """+str(nIter)+"""\n"""
    print params
             
    cProfile.runctx('lcap.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)',globals(),locals(),'Profile.prof')
    s = pstats.Stats('Profile.prof')
    print '---------------Python based LCA---------------'
    s.strip_dirs().sort_stats('time').print_stats()

    cProfile.runctx('lcac.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)',globals(),locals(),'Profile.prof')
    s = pstats.Stats('Profile.prof')
    print '---------------Cython based LCA---------------'
    s.strip_dirs().sort_stats('time').print_stats()

    cProfile.runctx('lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)',globals(),locals(),'Profile.prof')
    s = pstats.Stats('Profile.prof')
    print '----------Vectorized Cython based LCA--------'
    s.strip_dirs().sort_stats('time').print_stats()

    dictsIn = np.array(dictsIn,order='F')
    stimuli = np.array(stimuli,order='F')
    coeffs = np.array(coeffs,order='F')
    batchCoeffs = np.array(batchCoeffs,order='F')
    thresh = np.array(thresh,order='F')

    cProfile.runctx('lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)',globals(),locals(),'Profile.prof')
    s = pstats.Stats('Profile.prof')
    print '---------------Fortran based LCA--------------'
    s.strip_dirs().sort_stats('time').print_stats()

if __name__ == '__main__':
    main()
