cimport cython
import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
def infer(np.ndarray[DTYPE_t,ndim=2] basis,np.ndarray[DTYPE_t,ndim=2] coeffs,np.ndarray[DTYPE_t,ndim=2] stimuli,double eta, double lamb,int nIter,int softThresh,double adapt):
    """Infers sparse coefficients for dictionary elements when representing a stimulus using LCA algorithm.

    Args:
        basis: Dictionary used to represent stimuli. Should be arranged along rows.
        coeffs: Values to start pre-threshold dictionary coefficients at for all stimuli.
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
    
    cdef unsigned int ii,jj,kk,numStim,numDict
    numStim = stimuli.shape[0]
    numDict = basis.shape[0]
    #Initialize u and s
    cdef np.ndarray[DTYPE_t,ndim=2] u = np.array([coeffs[ii] for ii in xrange(stimuli.shape[0])], dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] s = np.zeros((numStim,numDict), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] b = np.zeros((numStim,numDict), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] ci = np.zeros((numStim,numDict), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] c = np.zeros((numDict,numDict), dtype = DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] thresh = np.ones(numStim,dtype = DTYPE)

    #Calculate c: overlap of basis functions with each other minus identity
    #Only calculate for elemets below the diagonal, then copies to above and leave diagonal at zero
    c = np.dot(basis,basis.T)-np.eye(numDict)
    #b[i,j] is the overlap from stimuli:i and basis:j
    b = np.dot(stimuli,basis.T)
    thresh = np.mean(np.absolute(b),axis=1)
    #Update u[i,j] and s[i,j] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated (s) times overlap with rest of basis
        ci = np.dot(s,c)
        #Update pre-threshold variables
        u[:] = eta*(b-ci)+(1-eta)*u
        #Threshold
        if softThresh == 1:
            s[:] = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis])
        else:
            s[:] = u
            s[np.absolute(s) < thresh[:,np.newaxis]] = 0.
        thresh[thresh > lamb] = adapt*thresh[thresh > lamb]
    return (s,u,thresh)

