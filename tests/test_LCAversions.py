from __future__ import print_function
import numpy as np

from LCApython import lca as lcap
from LCAcythonv import lca as lcav
from LCAnumbaprog import lca as lcag

def setup__module():
    pass

def teardown_module():
    pass

class test_infer():

    def setup(self):
        self.num = 64
        self.nIter = 300
        self.dictionary = np.diag(np.ones(self.num))
        self.coeffs = np.zeros(shape=(self.num,self.num))
        self.stimuli = np.diag(np.ones(self.num))
        self.eta = .1
        self.lamb = .5
        self.softThresh = int(0)
        self.adapt = .99

    def teardown(self):
        pass

    def test_cythonv(self):
        self.s,self.u,self.thresh = lcav.infer(self.dictionary,self.coeffs,self.stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(self.s,np.diag(np.ones(self.num)))
        assert np.allclose(self.u,np.diag(np.ones(self.num)))


    def test_numbaprog(self):
        self.dictionary = np.array(np.diag(np.ones(self.num)),dtype=np.float32,order='F')
        self.coeffs = np.array(np.diag(np.zeros(self.num)),dtype=np.float32,order='F')
        self.stimuli = np.array(np.diag(np.ones(self.num)),dtype=np.float32,order='F')
        
        self.s,self.u,self.thresh = lcag.infer(self.dictionary,self.coeffs,self.stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(self.s,np.diag(np.ones(self.num)))
        assert np.allclose(self.u,np.diag(np.ones(self.num)))
        

    def test_python(self):
        self.s,self.u,self.thresh = lcap.infer(self.dictionary,self.coeffs,self.stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt)
        assert np.allclose(self.s,np.diag(np.ones(self.num)))
        assert np.allclose(self.u,np.diag(np.ones(self.num)))
