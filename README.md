# FE<sup>2</sup> for multiscale analysis

FE<sup>2</sup> is a Finite Element Analysis (FEA) software aimed for performing multiscale analysis in which multiple finite element models are
solved in a nested scheme. Such code is built with the [JemJive](https://dynaflow.com/) numerical toolkit and highly influenced by the 
[prnn3d](https://github.com/SLIMM-Lab/prnn3d) C++ code.

Examples and results can be found in **`examples/`** for 2D and 3D cases.

Unlike [prnn3d](https://github.com/SLIMM-Lab/prnn3d), this implementation features:
* Multithreading with `pthread`
* Solve a Representative Volume Element (RVE) subject to Periodic Boundary Condition (PBC) without the necessity of setting a complete FE<sup>2</sup> framework
* Macro strain fields can be directly imposed on a single RVE, for instance, macro strain fields can be a function of the time step. This facilitates data generation and training

TODO:
* Parallel computing with Message Passing Interface (MPI)

An optional [online PRNN material using LibTorch](docs/prnn.md) can load
pretrained pyprnn weights and replace the macro FE2 material. Its J2 histories
follow Jive's accepted/rejected increment lifecycle.
