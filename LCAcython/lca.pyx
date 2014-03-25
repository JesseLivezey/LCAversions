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
@cython.boundscheck(False)
@cython.wraparound(False)
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
    
    assert basis.dtype == DTYPE and coeffs.dtype == DTYPE and stimuli.dtype == DTYPE
    cdef unsigned int ii,jj,kk
    cdef double thresh
    #Initialize u and s
    cdef np.ndarray[DTYPE_t,ndim=2] u = np.array([coeffs[ii] for ii in xrange(stimuli.shape[0])], dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] s = np.zeros((stimuli.shape[0],basis.shape[0]), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] b = np.zeros((stimuli.shape[0],basis.shape[0]), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] ci = np.zeros(basis.shape[0], dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] c = np.zeros((basis.shape[0],basis.shape[0]), dtype = DTYPE)

    #Calculate c: overlap of basis functions with each other minus identity
    #Only calculate for elemets below the diagonal, then copies to above and leave diagonal at zero
    for ii in xrange(basis.shape[0]):
        for jj in xrange(ii):
            c[ii,jj] = np.dot(basis[ii],basis[jj])
            c[jj,ii] = c[ii,jj]
    #b[i,j] is the overlap from stimuli:i and basis:j
    b = np.dot(stimuli,basis.T)
    #Loop over stimuli
    for ii in xrange(stimuli.shape[0]):
        thresh = np.mean(np.absolute(b))
        #Update u[i] and s[i] for nIter time steps
        for kk in xrange(nIter):
            #Calculate ci: amount other neurons are stimulated (s) times overlap with rest of basis
            ci = np.dot(s[ii],c)
	    #for jj,base in enumerate(basis):
	    #    ci[jj] = np.dot(c[jj,:,s[ii,:])
	    #Running the u[i], s[i] updates as vector operations takes longer than the loop
            u[ii] = eta*(b[ii]-ci)+(1-eta)*u[ii]
            for jj in xrange(basis.shape[0]):
                #u[ii,jj] = eta*(b[ii,jj]-ci[jj])+(1-eta)*u[ii,jj]
                s[ii,jj] = thresholdF(u[ii,jj],thresh,softThresh)
            if thresh > lamb:
                thresh = adapt*thresh
    return (s,u,thresh)

