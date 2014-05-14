from __future__ import print_function
import numpy as np

from LCAversions.LCAnumpy import lca as lcan
from LCAversions.LCAcythonv import lca as lcav
from LCAversions.LCAnumbaprog import lca as lcag
from LCAversions.LCAfortran import lca as lcaf

def setup__module():
    pass

def teardown_module():
    pass

class test_infer():

    def setup(self):
        self.rng = np.random.RandomState(0)
        self.num = 64
        self.numDict = 1024
        self.numStim = 64
        self.dataSize = 128
        self.nIter = 500
        self.eta = .05
        self.lamb = .05
        self.softThresh = int(0)
        self.adapt = .1

    def teardown(self):
        pass

    def test_cythonv(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        s,u,thresh = lcav.infer(dictionary,coeffs,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        coeffs = np.zeros(shape=(self.numStim,self.numDict))
        s,u,thresh = lcav.infer(dictionary,coeffs,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)

    def test_numbaprog(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,dtype=np.float32,order='F')
        coeffs = np.array(coeffs,dtype=np.float32,order='F')
        stimuli = np.array(stimuli,dtype=np.float32,order='F')
        s,u,thresh = lcag.infer(dictionary,coeffs,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        coeffs = np.zeros(shape=(self.numStim,self.numDict))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,dtype=np.float32,order='F')
        coeffs = np.array(coeffs,dtype=np.float32,order='F')
        stimuli = np.array(stimuli,dtype=np.float32,order='F')
        s,u,thresh = lcag.infer(dictionary,coeffs,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)

    def test_fortran(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,order='F')
        coeffs = np.array(coeffs,order='F')
        stimuli = np.array(stimuli,order='F')
        s = np.zeros_like(coeffs,order='F')
        u = np.zeros_like(coeffs,order='F')
        thresh = np.zeros(self.num,order='F')
        lcaf.lca(dictionary,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt,s,u,thresh,self.num,self.num,self.num)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        coeffs = np.zeros(shape=(self.numStim,self.numDict))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,order='F')
        coeffs = np.array(coeffs,order='F')
        stimuli = np.array(stimuli,order='F')
        s = np.zeros_like(coeffs,order='F')
        u = np.zeros_like(coeffs,order='F')
        thresh = np.zeros(self.numStim,order='F')
        lcaf.lca(dictionary,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt,s,u,thresh,self.numDict,self.numStim,self.dataSize)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)

    def test_numpy(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        s,u,thresh = lcan.infer(dictionary,coeffs,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        coeffs = np.zeros(shape=(self.numStim,self.numDict))
        s,u,thresh = lcan.infer(dictionary,coeffs,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)
