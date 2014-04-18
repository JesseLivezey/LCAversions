import numpy as np

#Initialize settings for inference
def infer(basis,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt):
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
        #for ii in xrange(stimuli.shape[0]):
        #    for jj in xrange(basis.shape[0]):
        #        s[ii,jj] = thresholdF(u[ii,jj],thresh,softThresh)
        for ii in xrange(numStim):
            if thresh[ii] > lamb:
                thresh[ii] = adapt*thresh[ii]
    return (s,u,thresh)

#Function to perform thresholding
def thresholdF(u,thresh,softThresh):
    if u < thresh and u > -thresh:
        sTemp = 0
    elif softThresh == 1:
        sTemp = u-np.sign(u)*thresh
    else:
        sTemp = u
    return sTemp
