import logging
import os
from pathlib import Path

import GTS as gts
from mastersproject.util.logging_util import __setup_logging

logger = logging.getLogger(__name__)


def test_create_grid():
    """Create a standard grid.

    Returns the ContactMechanicsBiot class with an overwritten create_grid method.
    """
    _this_file = Path(os.path.abspath(__file__)).parent
    _results_path = _this_file / "results"
    _results_path.mkdir(parents=True, exist_ok=True)  # Create path if not exists
    __setup_logging(_results_path)
    logger.info(f"Path to results: {_results_path}")

    # --- DOMAIN ARGUMENTS ---
    params = {
        "mesh_args": {"mesh_size_frac": 10, "mesh_size_min": 10, "mesh_size_bound": 10},
        "bounding_box": {
            "xmin": -6,
            "xmax": 80,
            "ymin": 55,
            "ymax": 150,
            "zmin": 0,
            "zmax": 50,
        },
        "shearzone_names": ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"],
    }

    gb = gts.isc_modelling.create_isc_domain(_results_path, **params, n_refinements=0)[
        0
    ]

    # Standard variables for ContactMechanicsISC class
    params["viz_folder_name"] = _results_path
    params["solver"] = "direct"
    params["stress"] = gts.isc_modelling.stress_tensor()
    params["length_scale"] = 1
    params["scalar_scale"] = 1

    cm = gts.isc_modelling.ContactMechanicsISCWithGrid(params, gb)

    return cm
