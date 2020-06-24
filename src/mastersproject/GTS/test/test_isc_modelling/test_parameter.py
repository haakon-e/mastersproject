from pathlib import Path

from GTS.isc_modelling.parameter import BiotParameters, stress_tensor, shearzone_injection_cell, GrimselGranodiorite


class TestFlowParameters:
    def test_validate_source_scalar_borehole_shearzone(self):
        assert False

    def test_set_fracture_aperture_from_cubic_law(self):
        here = Path(__file__).parent / "simulations"
        _sz = 6
        params = BiotParameters(
            folder_name=here,
            stress=stress_tensor(),
            injection_rate=1 / 6,
            frac_permeability=1e-9,
            intact_permeability=1e-12,
            well_cells=shearzone_injection_cell,
            rock=GrimselGranodiorite(),
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,
            },
        )
        assert params.initial_fracture_aperture is not None
