from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csc_matrix

import porepy as pp
from refinement.grid_refinement import (
    coarse_fine_cell_mapping,
    gb_coarse_fine_cell_mapping,
    refine_mesh_by_splitting,
)


class TestGridRefinement:
    def test_refine_mesh_by_splitting(self):
        # Create initial grid
        path_head = "TestGridRefinement/test_refine_mesh_by_splitting"
        n_grids = 3
        _, file_name, gb = create_gb_with_simple_fracture(path_head=path_head)
        in_file = f"{file_name}.geo"
        out_file = file_name

        # Run refine_mesh_by_splitting
        gb_generator = refine_mesh_by_splitting(in_file, out_file, dim=3)
        gb_list = [next(gb_generator) for _ in range(0, n_grids)]
        gb_generator.close()

        # Assertions
        assert len(gb_list) == n_grids, "Create as many grids as expected"
        assert (
            gb.num_cells() == gb_list[0].num_cells()
        ), "The first grid should correspond to the initial grid"
        num_cells = [_gb.num_cells() for _gb in gb_list]
        for i in range(1, n_grids):
            assert (
                num_cells[i - 1] < num_cells[i]
            ), "Each subsequent grid should be a refinement of the previous grid"

    def test_coarse_fine_cell_mapping(self):
        # Create 2 grid buckets
        path_head = "TestGridRefinement/test_coarse_fine_cell_mapping"
        gb, gb_ref = generate_n_grids(path_head, n_grids=2)

        # For the 2d-grids, create a coarse_fine cell mapping
        g: pp.Grid = gb.grids_of_dimension(2)[0]
        g_ref: pp.Grid = gb_ref.grids_of_dimension(2)[0]
        coarse_fine = coarse_fine_cell_mapping(g, g_ref)

        # Assertions
        assert (
            coarse_fine.indices.size == g_ref.num_cells
        ), "Every fine cell should be inside exactly one coarse cell"

    def test_gb_coarse_fine_cell_mapping(self):
        # Create 2 grid buckets
        path_head = "TestGridRefinement/test_gb_coarse_fine_cell_mapping"
        gb_list = generate_n_grids(path_head, n_grids=2)
        gb, gb_ref = gb_list[0], gb_list[1]

        # Create coarse_fine cell mappings for each grid in grid bucket
        gb_coarse_fine_cell_mapping(gb, gb_ref)

        # Assertions
        for g, data in gb:
            key = "coarse_fine_cell_mapping"
            assert key in data, "Key should be assigned to each grid dictionary"
            assert isinstance(data[key], csc_matrix)

            # Check the shape of the csc matrix
            node_num = data["node_number"]
            g_ref_lst = gb_ref.get_grids(
                lambda g: gb_ref.node_props(g, "node_number") == node_num
            )
            assert len(g_ref_lst) == 1, "There should be exactly one corresponding grid"
            g_ref = g_ref_lst[0]

            coarse_fine = data[key]
            assert (
                coarse_fine.shape[0] == g_ref.num_cells
            ), "Check the shape of the coarse_fine cell mapping"
            assert (
                coarse_fine.shape[1] == g.num_cells
            ), "Check the shape of the coarse_fine cell mapping"


def generate_n_grids(path_head: str, n_grids: int) -> List[pp.GridBucket]:
    network, file_name, gb = create_gb_with_simple_fracture(path_head=path_head)
    in_file = f"{file_name}.geo"
    out_file = file_name

    gb_generator = refine_mesh_by_splitting(in_file, out_file, dim=3)
    gb_list = [next(gb_generator) for _ in range(0, n_grids)]
    gb_generator.close()
    return gb_list


def create_gb_with_simple_fracture(
    path_head: str,
) -> Tuple[pp.FractureNetwork3d, Path, pp.GridBucket]:
    """ Setup method.

    Set up a fracture network and mesh it.

    Parameters
    ----------
    path_head : str
        head of path to store test files

    Returns
    -------
    network : pp.FractureNetwork3d
    file_name : Path
        path to the gmsh files
        with 'path/to/file_name', but without .geo/.msh endings
    gb : pp.GridBucket
    """
    path = Path(__file__).resolve().parent / "results"
    root = path / path_head
    root.mkdir(parents=True, exist_ok=True)
    file_name = root / "gmsh_frac_file"

    f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
    domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
    network = pp.FractureNetwork3d([f_1], domain=domain)
    # mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 2, "mesh_size_min": 0.1}
    mesh_args = {"mesh_size_bound": 10, "mesh_size_frac": 10, "mesh_size_min": 10}

    gb = network.mesh(mesh_args=mesh_args, file_name=str(file_name))
    return network, file_name, gb
