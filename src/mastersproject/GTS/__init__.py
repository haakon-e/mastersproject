# Import new model data
from GTS.ISC_data.isc import (
    ISCData,  # Data set
    swiss_to_gts,  # Transformation
    borehole_to_global_coords,  # Transformation
)

# Import fracture tools
from GTS.ISC_data.fracture import (
    convex_plane,
    fracture_network,
)

# Plane fit tools
from GTS.fit_plane import (
    plane_from_points,
    convex_hull,
)

# -------------------------
# --- SETUPS AND MODELS ---
# -------------------------

from GTS.isc_modelling.parameter import stress_tensor

# --- SETUP AND RUN MODEL---
from GTS.isc_modelling.setup import (
    run_mechanics_model,
    run_biot_model,
    run_biot_gts_model,
    run_abstract_model,
)

# --- MODELS ---

# Contact mechanics model
from GTS.isc_modelling.mechanics import ContactMechanicsISC

# Contact Mechanics Biot
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotISC

# Flow model
from GTS.isc_modelling.flow import (
    Flow,
    FlowISC,
)

# PREPARE DIRECTORIES AND PARAMETERS
from GTS.isc_modelling.setup import (
    prepare_directories,
    SetupParams,
)
