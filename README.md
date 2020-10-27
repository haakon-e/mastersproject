# Simulation of injection-induced deformation for a stimulation experiment at the Grimsel underground laboratory
Master of Science Thesis in Applied and Computational Mathematics

Link to thesis [coming]

To run the code, clone the repository, install all dependencies, and add the `GTS` folder to importable modules.
Access the python console and run

```python
from GTS.isc_modelling.runscripts.hs1_stimulation import box_runscript

time_machine = box_runscript(case="A1")  # Replace by A2, B1, B2 for the other cases.
```

The code base is extremely flexible in the sense that the components for the coupled PDE are modularized, allowing for arbitrary extensions to the existing setup. 
Separate and independent classes handle injection phases, timestepping, gridding, etc.
A class, `TimeMachine`, assembles and executes the simulation for the given input parameters. Simulation results are automatically exported to a format compatible with ParaView.
