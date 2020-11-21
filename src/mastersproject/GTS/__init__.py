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

# --- MODELS ---

# Contact mechanics model
from GTS.isc_modelling.mechanics import Mechanics

# Flow model
from GTS.isc_modelling.flow import (
    Flow,
    FlowISC,
)

# Contact Mechanics Biot model
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotBase
from GTS.isc_modelling.isc_model import ISCBiotContactMechanics

# PARAMETERS
from GTS.isc_modelling.parameter import (
    GrimselGranodiorite,
    BaseParameters,
    GeometryParameters,
    MechanicsParameters,
    FlowParameters,
    BiotParameters,
)

__all__ = [
    "ISCData",
    "swiss_to_gts",
    "borehole_to_global_coords",
    "convex_plane",
    "fracture_network",
    "plane_from_points",
    "convex_hull",
    "stress_tensor",
    "Mechanics",
    "Flow",
    "FlowISC",
    "ContactMechanicsBiotBase",
    "ISCBiotContactMechanics",
    "GrimselGranodiorite",
    "BaseParameters",
    "GeometryParameters",
    "MechanicsParameters",
    "FlowParameters",
    "BiotParameters",
]
