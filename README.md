# Simulation of injection-induced deformation for a stimulation experiment at the Grimsel underground laboratory
Master of Science Thesis in Applied and Computational Mathematics

Link to thesis: http://bora.uib.no/handle/1956/24463

## Installation instructions

Conda is recommended to set up the dependencies for this project. Install either Anaconda or Miniconda, 
visit their website and follow the appropriate instructions.

The algorithms and solvers in the project can be accelerated by using Intel distributions of numpy, scipy, and python.
If you do not have an Intel processor on your computer, you may skip the following steps.
```
conda update conda
conda create -n idp -c intel intelpython3_full python=3.7
conda activate idp
conda config --add channels conda-forge
conda config --add channels intel
```

Install the project dependencies. Some packages are install via pip since they seem to be incompatible with the 
intel python setup.
```
conda install pendulum pydantic meshio shapely cython future gmsh
python -m pip install networkx vtk sympy
```

The direct solver is accelerated with [PyPardiso](https://github.com/haasad/PyPardisoProject). To install, run
```
conda install -c haasad pypardiso
```
Additional packages for development and testing:
```
conda install pytest pytest-cov pytest-mock black=20.8b1 flake8 isort
```
See [PorePy Installation](https://github.com/pmgbergen/porepy/blob/develop/Install.md) for additional dependencies and 
instructions to install PorePy. Then, clone this repository. Add PorePy and this project to conda 
(See [this link](https://stackoverflow.com/a/59903590/12580152) for information about `conda develop`):
```
conda develop path/to/porepy/src
conda develop path/to/mastersproject/src
conda develop path/to/mastersproject/src/mastersproject
```

To run the code, clone the repository, install all dependencies, and add the `GTS` folder to importable modules.
Install PorePy by following the instructions on their [GitHub page](https://github.com/pmgbergen/porepy/tree/1.2.1). Currently, [PorePy version 1.2.1](https://github.com/pmgbergen/porepy/releases/tag/1.2.1) is supported.
Access the python console and run

```python
from GTS.isc_modelling.runscripts.hs1_stimulation import box_runscript

time_machine = box_runscript(case="A1")  # Replace by A2, B1, B2 for the other cases.
```

The code base is extremely flexible in the sense that the components for the coupled PDE are modularized, allowing for arbitrary extensions to the existing setup. 
Separate and independent classes handle injection phases, timestepping, gridding, etc.
A class, `TimeMachine`, assembles and executes the simulation for the given input parameters. Simulation results are automatically exported to a format compatible with ParaView.
