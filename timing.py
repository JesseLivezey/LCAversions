#This file will time various versions of LCA
from __future__ import division
import numpy as np
from timeit import default_timer as timer

from LCAnumpy import lca as lcan
from LCAfortran import lca as lcaf
from LCAcythonv import lca as lcav
#from LCAnumbaproc import lca as lcan
from LCAnumbaprog import lca as lcag

def main():
    """Profiles various versions of LCA."""

    nshort = 6
    tshort = 2
    nmed = 3
    tmed = 6
    nlong = 1
    
    #Setup variables for inference
    numDict = int(1024)
    numBatch = int(128)
    dataSize = int(256)
    dictsIn = np.random.randn(numDict,dataSize)
    coeffs = np.random.randn(numBatch,numDict)
    stimuli = np.random.randn(numBatch,dataSize)
    batchCoeffs = np.random.randn(numBatch,numDict)
    eta = .01
    lamb = .05
    nIter = 150
    softThresh = int(1)
    adapt = .99
    thresh = np.random.randn(numBatch)
    
    #LCA
    params = """Parameters:
             numDict: """+str(numDict)+"""
             numBatch: """+str(numBatch)+"""
             dataSize: """+str(dataSize)+"""
             nIter: """+str(nIter)+"""\n"""
    print params
             
    start = timer()
    lcan.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcan.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '---------------Numpy based LCA----------------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt

    start = timer()
    lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '----------Vectorized Cython based LCA---------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt
    

    dictsIn = np.array(dictsIn,order='F')
    stimuli = np.array(stimuli,order='F')
    coeffs = np.array(coeffs,order='F')
    batchCoeffs = np.array(batchCoeffs,order='F')
    thresh = np.array(thresh,order='F')

    start = timer()
    lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '---------------Fortran based LCA--------------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt

    """
    start = timer()
    lcag.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcag.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '-------------Numbapro CPU based LCA-----------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt
    """

    start = timer()
    lcag.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcag.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '----------------GPU based LCA-----------------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt

if __name__ == '__main__':
    main()
