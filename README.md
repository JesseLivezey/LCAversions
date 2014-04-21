LCAversions
===========

Locally Competitive Algorithm written in Python using various packages.

Current implementations include:

* LCApython: Pure python+numpy. Requires numpy.
* LCAcythonv(float64 only): Cython with minibatch vectorization. Requires numpy, cython.
* LCAfortan: Fortran 90 with python wrapper. Requires f2py.
* LCAnumbaprog(float32 only): NumbaPro GPU implementation. Requires numbapro.

Cython based version can be compiled using the command:
```
python setup.py build_ext --inplace
```
in the individual folders.

The Fortran based version can be compiled using the command:
```
f2py -c -m lca lca.f90
```
in the fortran folder.

To run tests, do:
```
nosetests
```
from the base directory.
