import numpy as np
from numbapro import cuda
import numbapro.cudalib.cublas as cublas
from numba import *

@cuda.jit('void(f4[:,:],f4[:,:])')
def uinit(u,coeffs):
    n = u.shape[0]
    m = u.shape[1]
    i,j = cuda.grid(2)
    
    u[i,j] = coeffs[i,j]

@cuda.jit('void(f4[:,:],f4[:,:])')
def cinit(dictionary,c):
    n = dictionary.shape[0]
    m = dictionary.shape[1]
    i,j = cuda.grid(2)
    
    if (i != j):
        for k in xrange(m):
            c[i,j] += dictionary[i,k]*dictionary[j,k]

@cuda.jit('void(f4[:,:])')
def csub(c):
    i = cuda.grid(1)
    
    c[i,i] = 0.

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def binit(dictionary,stimuli,b):
    n = stimuli.shape[0]
    m = dictionary.shape[0]
    k = dictionary.shape[1]
    i,j = cuda.grid(2)

    for r in xrange(k):
        b[i,j] += stimuli[i,r]*dictionary[j,r]

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4[:],f4,f4,i4)')
def iterate(c,b,ci,u,s,eta,thresh,lamb,adapt,softThresh):
    n = u.shape[0]
    m = u.shape[1]
    i,j = cuda.grid(2)
    
    u[i,j] = eta*(b[i,j]-ci[i,j])+(1-eta)*u[i,j]
    if u[i,j] < thresh[i] and u[i,j] > -thresh[i]:
        s[i,j] = 0.
    elif softThresh == 1:
        if u[i,j] > 0.:
            s[i,j] = u[i,j]-thresh[i]
        else:
            s[i,j] = u[i,j]+thresh[i]
    else:
        s[i,j] = u[i,j]
    if thresh[i] > lamb:
        thresh[i] = thresh[i]*lamb

def infer(dictionary,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt):
#Get Blas routines
    bs = cublas.Blas()
#Initialize arrays
    numDict = dictionary.shape[0]
    numStim = stimuli.shape[0]
    dataLength = stimuli.shape[1]
    d_u = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_s = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_b = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_ci = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_c = cuda.to_device(np.zeros((numDict,numDict),dtype=np.float32,order='F'))
    
    #Move inputs to GPU
    d_dictionary = cuda.to_device(np.array(dictionary,dtype=np.float32,order='F'))
    d_coeffs = cuda.to_device(np.array(coeffs,dtype=np.float32,order='F'))
    d_stimuli = cuda.to_device(np.array(stimuli,dtype=np.float32,order='F'))

    blockdim2 = (32,32)
    blockdim1 = 32
    #griddimc = (int(numDict/blockdim[0]),int(numDict/blockdim[1]))
    griddimcsub = int(numDict/blockdim1)
    griddimb = (int(numStim/blockdim2[0]),int(numDict/blockdim2[1]))
    griddimi = (int(numStim/blockdim2[0]),int(numDict/blockdim2[1]))
    
    #Calculate c: overlap of basis functions with each other minus identity
    #cinit[griddimc,blockdim](d_dictionary,d_c)
    bs.gemm('N','T',numDict,numDict,dataLength,1.,d_dictionary,d_dictionary,0.,d_c)
    csub[griddimcsub,blockdim1](d_c)
    #binit[griddimb,blockdim2](d_dictionary,d_stimuli,d_b)
    bs.gemm('N','T',numStim,numDict,dataLength,1.,d_stimuli,d_dictionary,0.,d_b)
    thresh = np.mean(np.absolute(d_b.copy_to_host()),axis=1)
    d_thresh = cuda.to_device(thresh)
    #Update u[i] and s[i] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated times overlap with rest of basis
        bs.gemm('N','N',numStim,numDict,numDict,1.,d_s,d_c,0.,d_ci)
        iterate[griddimi,blockdim2](d_c,d_b,d_ci,d_u,d_s,eta,d_thresh,lamb,adapt,softThresh)
    u = d_u.copy_to_host()
    s = d_s.copy_to_host()
    return (s,u,thresh)
