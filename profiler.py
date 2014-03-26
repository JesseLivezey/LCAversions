#This file will time various versions of LCA
from __future__ import division
import numpy as np
from timeit import default_timer as timer

from LCApython import lca as lcap
from LCAfortran import lca as lcaf
from LCAcython import lca as lcac
from LCAcythonv import lca as lcav
from LCAnumbaproc import lca as lcan
from LCAnumbaprog import lca as lcag

def main():
    """Profiles functions related to the network."""

    nshort = 6
    tshort = 1
    nmed = 3
    tmed = 5
    nlong = 1
    
    #Setup variables for inference
    numDict = int(800)
    numBatch = int(128)
    dataSize = int(256)
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
             
    """
    start = timer()
    lcap.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            lcap.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            lcap.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            lcap.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '---------------Python based LCA---------------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt
    """

    """
    start = timer()
    lcac.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            lcac.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            lcac.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            lcac.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '---------------Cython based LCA---------------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt
    """


    """
    start = timer()
    lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            lcav.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '----------Vectorized Cython based LCA---------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt
    """

    """
    dictsIn = np.array(dictsIn,order='F')
    stimuli = np.array(stimuli,order='F')
    coeffs = np.array(coeffs,order='F')
    batchCoeffs = np.array(batchCoeffs,order='F')
    thresh = np.array(thresh,order='F')

    start = timer()
    lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '---------------Fortran based LCA--------------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt
    """

    """
    start = timer()
    lcan.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            lcan.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            lcan.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            lcan.infer(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '-------------Numbapro CPU based LCA-----------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt
    """


    #dictsIn = np.array(dictsIn,dtype)
    #stimuli = np.array(stimuli,order='F')
    #coeffs = np.array(coeffs,order='F')
    
    start = timer()
    lcag.lca(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
    dt = timer()-start
    if dt < tshort:
        for ii in xrange(nshort-1):
            start = timer()
            lcag.lca(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nshort
        dt = dt/(nshort)
    elif dt < tmed:
        for ii in xrange(nmed-1):
            start = timer()
            lcag.lca(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nmed
        dt = dt/(nmed)
    else:
        for ii in xrange(nlong-1):
            start = timer()
            lcag.lca(dictsIn,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt)
            dt = dt+timer()-start
        num = nlong
        dt = dt/(nlong)
    print '----------------GPU based LCA-----------------'
    print 'Average time over '+str(num)+' trials:'
    print '%f s' % dt

if __name__ == '__main__':
    main()
