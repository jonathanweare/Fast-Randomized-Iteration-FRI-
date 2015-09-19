# Fast-Randomized-Iteration-FRI-
A simple example in C++ of FRI applied to computing the dominant eigenvalue of a matrix.


---
#### Algorithm

[Fast Randomized Iteration (FRI)](http://arxiv.org/abs/1207.2866) is a recent randomization approach to certain problems in numerical linear algebra proposed by [Lek-Heng Lim](http://www.stat.uchicago.edu/~lekheng/) and [Jonathan Weare](http://www.stat.uchicago.edu/~weare/). It is useful for eigenproblems, linear system solves, and matrix exponentiation, for example, in dimensions so large that even the solution vector itself cannot be stored.  In the constant cost (in the size of the matrix) per iteration variant exemplified here it estimates low dimensional projections of the solution rather than the full solution vector.

---
#### This Code

This repo provides a simple C++11 implementation of an FRI algorithm to find the dominant eigenvector of the transfer matrix associated with the 2 dimensional Ising model.  The matrix is roughly of size 10<sup>15</sup> x 10<sup>15</sup>.  This is the first example presented in the Lim and Weare paper.  The code is designed to be relatively easy to parse and is far from optimized for performance.  It consists of one driver file for this particular example, one file containing genereric routines of the type required by any FRI implementation, and one header file.

Along with the algorithm code, the repo also contains a copy of the Lim and Weare ArXiv paper introducing Fast Randomized Iteration. Try the example and experiment.  Comments welcome!

If you use the code please cite this repository using the DOI above.
