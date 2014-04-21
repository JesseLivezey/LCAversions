#
#
# Jesse Livezey 2014-04-19
#


import numpy as np

#Initialize settings for inference
def infer(basis,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt):
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
    numDict = basis.shape[0]
    numStim = stimuli.shape[0]
    dataSize = basis.shape[1]
    #Initialize u and s
    u = np.array([coeffs[ii] for ii in xrange(numStim)])
    s = np.zeros((numStim,numDict))
    b = np.zeros((numStim,numDict))
    ci = np.zeros((numStim,numDict))
    c = np.zeros((numDict,numDict))
    #Calculate c: overlap of basis functions with each other minus identity
    #should use symmetry to cut back on time, probably not important
    for ii in xrange(numDict):
        for jj in xrange(ii):
            c[ii,jj] = np.dot(basis[ii],basis[jj])
            c[jj,ii] = c[ii,jj]
    #b[i,j] is the overlap fromstimuli:i and basis:j
    b = np.dot(stimuli,basis.T)
    thresh = np.mean(np.absolute(b),axis=1)
    #Update u[i] and s[i] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated times overlap with rest of basis
        ci = np.dot(s,c)
        u = eta*(b-ci)+(1-eta)*u
        if softThresh == 1:
            s = np.sign(u)*np.maximum(0.,np.absolute(u)-np.tile(np.array([thresh]).T,(1,numDict))) 
        else:
            s = np.sign(u)*(np.maximum(0.,np.absolute(u)-np.tile(np.array([thresh]).T,(1,numDict)))+np.greater(np.absolute(u),np.tile(np.array([thresh]).T,(1,numDict))).astype(np.float64)*np.tile(np.array([thresh]).T,(1,numDict)))
        for ii in xrange(numStim):
            if thresh[ii] > lamb:
                thresh[ii] = adapt*thresh[ii]
    return (s,u,thresh)
