# Overview

Information about the In-Situ Stimulation and Circulation experiment at the Grimsel Test Site can be found here: http://www.grimsel.com/gts-phase-vi/isc-in-situ-stimulation-circulation-experiment

## Data extraction
The class which extracts raw data from the In-Situ Stimulation and Circulation (ISC) 
experiment at the Grimsel Test Site (GTS) is found in `GTS/ISC_data/isc.py`.
The raw data is loacted in `GTS/01BasicInputData`.

# Model setup

The most advanced model currently being tested is Biot's equations with contact mechanics for fractured porous media.

The setup bridges the ISC data with the porepy simulation framework (see: https://github.com/pmgbergen/porepy).
