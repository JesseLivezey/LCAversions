cimport cython
import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t

#Function to perform thresholding
@cython.profile(False)
cdef inline double thresholdF(double u,double thresh,int softThresh):
    cdef double sTemp
    if u < thresh and u > -thresh:
        sTemp = 0
    elif softThresh == 1:
        sTemp = u-np.sign(u)*thresh
    else:
        sTemp = u
    return sTemp

#Initialize settings for inference
@cython.wraparound(False)
@cython.boundscheck(False)
def infer(np.ndarray[DTYPE_t,ndim=2] basis,np.ndarray[DTYPE_t,ndim=2] coeffs,np.ndarray[DTYPE_t,ndim=2] stimuli,double eta, double lamb,int nIter,int softThresh,double adapt):
    """Infers sparse coefficients for dictionary elements when representing a stimulus.

    Args:
        basis: Dictionary used to represent stimuli. Should be arranged along rows.
        coeffs: Values to start sparse coefficients at for all stimuli.
        stimuli: Goals for dictionary representation. Should be arranged along rows.
        eta: Controls rate of inference.
        thresh: Threshold used in calculation of output variable of model neuron.
        lamb: Minimum value for thresh.
        nIter: Numer of times to run inference loop.
        softThresh: Boolean choice of threshold type.
        adapt: Amount to change thresh by per run.

    Results:
        s: Post-threshold dictionary coefficients.
        u: Pre-threshold internal *voltage.*
        thresh: Final value of thresh variable.
        
    Raises:
    """
    
    cdef unsigned int ii,jj,kk,numberS,numberB
    nStimuli = stimuli.shape[0]
    nBasis = basis.shape[0]
    #Initialize u and s
    cdef np.ndarray[DTYPE_t,ndim=2] u = np.array([coeffs[ii] for ii in xrange(stimuli.shape[0])], dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] s = np.zeros((nStimuli,nBasis), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] b = np.zeros((nStimuli,nBasis), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] ci = np.zeros((nStimuli,nBasis), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] c = np.zeros((nBasis,nBasis), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thresh = np.ones(nStimuli,dtype = DTYPE)

    #Calculate c: overlap of basis functions with each other minus identity
    #Only calculate for elemets below the diagonal, then copies to above and leave diagonal at zero
    for ii in xrange(nBasis):
        for jj in xrange(ii):
            c[ii,jj] = np.dot(basis[ii],basis[jj])
            c[jj,ii] = c[ii,jj]
    #b[i,j] is the overlap from stimuli:i and basis:j
    b = np.dot(stimuli,basis.T)
    thresh = np.mean(np.absolute(b))*thresh
    #Update u[i,j] and s[i,j] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated (s) times overlap with rest of basis
        ci = np.dot(s,c)
        u = eta*(b-ci)+(1-eta)*u
        for ii in xrange(nStimuli):
            for jj in xrange(nBasis):
                #s[ii,jj] = thresholdF(u[ii,jj],thresh[ii],softThresh)
                if u[ii,jj] < thresh[ii] and u[ii,jj] > -thresh[ii]:
                    s[ii,jj] = 0.
                elif softThresh == 1:
                    s[ii,jj] = u[ii,jj]-np.sign(u[ii,jj])*thresh[ii]
                else:
                    s[ii,jj] = u[ii,jj]
        for ii in xrange(nStimuli):
            if thresh[ii] > lamb:
                thresh[ii] = adapt*thresh[ii]
    return (s,u,thresh)

