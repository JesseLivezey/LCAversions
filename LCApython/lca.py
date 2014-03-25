import numpy as np

#Initialize settings for inference
def infer(basis,coeffs,stimuli,eta,lamb,nIter,softThresh,adapt):
    #Initialize u and s
    u = np.array([coeffs[ii] for ii in xrange(stimuli.shape[0])])
    s = np.zeros((stimuli.shape[0],basis.shape[0]))
    b = np.zeros((stimuli.shape[0],basis.shape[0]))
    ci = np.zeros((stimuli.shape[0],basis.shape[0]))
    c = np.zeros((basis.shape[0],basis.shape[0]))
    #Calculate c: overlap of basis functions with each other minus identity
    #should use symmetry to cut back on time, probably not important
    for ii in xrange(basis.shape[0]):
        for jj in xrange(ii):
            c[ii,jj] = np.dot(basis[ii],basis[jj])
            c[jj,ii] = c[ii,jj]
    #b[i,j] is the overlap fromstimuli:i and basis:j
    b = np.dot(stimuli,basis.T)
    thresh = np.mean(np.absolute(b))
    #Update u[i] and s[i] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated times overlap with rest of basis
        ci = np.dot(s,c)
        u = eta*(b-ci)+(1-eta)*u
        for ii in xrange(stimuli.shape[0]):
            for jj in xrange(basis.shape[0]):
                s[ii,jj] = thresholdF(u[ii,jj],thresh,softThresh)
        if thresh > lamb:
            thresh = adapt*thresh
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
