import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Union, Dict, Optional

import gmsh
import numpy as np
import pandas as pd
import porepy as pp
import vg
from GTS import ISCBiotContactMechanics
from GTS.isc_modelling.parameter import BiotParameters
from mastersproject.util.logging_util import timer
from porepy.fracs.fracture_importer import dfm_from_gmsh

logger = logging.getLogger(__name__)


class ISCBoxModel(ISCBiotContactMechanics):
    def __init__(self, params: BiotParameters, lcin: float, lcout: float):
        super().__init__(params)

        # See create_grid for details
        self.frac_num_map = None

        # characteristic mesh size within and outside fractured zone
        self.lcin = lcin
        self.lcout = lcout

    # --- Grid methods ---

    @timer(logger, "INFO")
    def create_grid(self, n_optimize=2, use_logger=True):
        """ Create GridBucket from fracture box model"""
        gb = create_grid(
            **self.params.dict(include={
                "bounding_box",
                "fraczone_bounding_box",
                "folder_name",
                "length_scale",
                "fractures",
            }),
            lcin=self.lcin,
            lcout=self.lcout,
            n_optimize_netgen=n_optimize,
            use_logger=use_logger,
        )
        self._gb = gb
        self.bounding_box = gb.bounding_box(as_dict=True)

        gb.set_node_prop(self._nd_grid(), key="name", val=self.params.intact_name)

        fgrids = self.gb.get_grids(lambda g: g.dim == self.Nd - 1)
        assert len(fgrids) == self.params.n_frac

    @property
    def gb(self) -> pp.GridBucket:
        return self._gb

    @gb.setter
    def gb(self, gb: pp.GridBucket):
        """ Set a grid bucket to the class"""
        self._gb = gb
        if gb is not None:
            self.bounding_box = gb.bounding_box(as_dict=True)
        # else:
        #     raise ValueError("Use create_grid instead")


def create_grid(
        bounding_box: dict,
        fraczone_bounding_box: dict,
        lcin: float,
        lcout: float,
        folder_name: Optional[Path] = None,
        length_scale: float = 1,
        fractures: Union[str, List[str]] = "all",
        n_optimize_netgen: int = 1,
        verbose: bool = False,
        run_gmsh_gui: bool = False,
        use_logger: bool = True,
) -> pp.GridBucket:
    """Create the ISC domain using the box model method

    Parameters
    ----------
    bounding_box : dict
        dictionary of 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'
        Bounding box of global domain
    fraczone_bounding_box : dict
        dictionary of 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'
        Bounding box of fractured zone
    lcin, lcout : float
        mesh size parameters within, and outside the fractured zone, respectively
        Should be unscaled (i.e.: They are automatically scaled by ls).
        For example: lcin=5, lcout=50
    folder_name : Path
        path to store the .msh file.
        If unset, .msh file will not be stored.
    length_scale : float
        length scale
    fractures : List of strings or {"all"}
        Which shearzones to mesh
    n_optimize_netgen : int
        number of times to optimize with the Netgen optimizer
    verbose : bool
        For debugging purposes: report on gmsh tags.
    run_gmsh_gui : bool
        For debugging purposes: display the mesh in gmsh gui.
    use_logger : bool
        Show output of grid generation log
    """
    if folder_name is None:
        tempdir: bool = True
        mshfile = None
    elif folder_name.is_dir():
        tempdir: bool = False
        mshfile = folder_name / "gmsh_frac_file.msh"
    else:
        raise ValueError("Path must be a directory")

    all_shearzones = ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]
    if fractures == "all":
        fractures = all_shearzones
    else:
        for s in fractures:
            assert s in all_shearzones, f"The shear zone {s} is not recognized!"

    # Constrain the mesh size parameter within the fractured zone
    lcin = lcin / length_scale
    lcout = lcout / length_scale

    # --- Initialize gmsh ----------------------------------------------------------------------------------------------

    gmsh.initialize()
    gmsh.model.add("isc-geometry")
    if use_logger:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.logger.start()
    kernel = gmsh.model.occ
    # Ask OpenCASCADE to compute more accurate bounding boxes of entities using the STL mesh:
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

    # Make bounding box for fractured zone
    xmin: float = fraczone_bounding_box.get("xmin") / length_scale
    ymin: float = fraczone_bounding_box.get("ymin") / length_scale
    zmin: float = fraczone_bounding_box.get("zmin") / length_scale
    xmax: float = fraczone_bounding_box.get("xmax") / length_scale
    ymax: float = fraczone_bounding_box.get("ymax") / length_scale
    zmax: float = fraczone_bounding_box.get("zmax") / length_scale

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    inner_box = kernel.addBox(xmin, ymin, zmin, dx, dy, dz)

    # Produce big bounding box. This has tag 13.
    gxmin: float = bounding_box.get("xmin") / length_scale
    gymin: float = bounding_box.get("ymin") / length_scale
    gzmin: float = bounding_box.get("zmin") / length_scale
    gdx: float = bounding_box.get("xmax") / length_scale - gxmin
    gdy: float = bounding_box.get("ymax") / length_scale - gymin
    gdz: float = bounding_box.get("zmax") / length_scale - gzmin
    outer_box = kernel.addBox(gxmin, gymin, gzmin, gdx, gdy, gdz)

    # Fragment the outer box with the inner box (to be fractured zone)
    # fmt: off
    out_volumes, map_volumes = kernel.fragment(
        [(3, outer_box)],
        [(3, inner_box)],
    )
    # fmt: on
    kernel.synchronize()

    if verbose:
        for o, c in zip(["outer_box", "inner_box"], map_volumes):
            logger.info(f"{o} -> mapped to tags {c}")
        logger.info("")
    # The second entry corresponds to the (first) fragment tool, i.e. the inner box (3, inner_box).
    # Fetch the new tag of this entity:
    inner_box = map_volumes[1]

    # Add 5 fractures
    data_path = Path(__file__).parent / "isc_frac_data.txt"
    data = pd.read_csv(data_path)
    fracs = []
    frac_names = []
    for i in data.index:
        xc, yc, zc = data.loc[i, ["x_c", "y_c", "z_c"]] / length_scale
        frac_name = data.loc[i, "shearzone"]  # Name of fracture
        if frac_name not in fractures:
            continue
        frac = kernel.addRectangle(xmin - dx, ymin - dy, zc, 3 * dx, 3 * dy)
        f = (2, frac)  # dimTag of fracture
        # ----- Rotate the fracture -----
        # The normal vector of the end result is stored in ["n_x", "n_y", "n_z"]
        # 1. Project this vector to the Rectangle (i.e.: nproj = [n_x, n_y, 0])
        n = data.loc[i, ["n_x", "n_y", "n_z"]].to_numpy(dtype=float)
        nproj = n.copy()
        nproj[-1] = 0
        # 2. Take the cross product of v and the initial rectangle normal vector
        # (i.e. v0 = [0, 0, 1]). This is the vector that defines the axis of revolution.
        # That is, this vector is the intersection of the original rectangle and the final
        # rectangle.
        v0 = np.array([0, 0, 1])
        x_cr, y_cr, z_cr = np.cross(v0, nproj)
        # 3. Find the angle between the vertical and the normal vector
        # angle = vg.angle(n, nproj, units="rad")
        angle = vg.angle(n, v0, units="rad")
        kernel.rotate([f], xc, yc, zc, x_cr, y_cr, z_cr, angle)

        fracs.append(f)
        frac_names.append(frac_name)

    # Fragment the box along the fracture surface
    # fmt: off
    out_frac, map_frac = kernel.fragment(
        inner_box,
        fracs,
    )
    # fmt: on
    kernel.synchronize()

    # Now remove all the surfaces (and their bounding entities) that are not on the
    # boundary of a volume, i.e. the parts of the cutting planes that "stick out" of
    # the volume:
    kernel.remove(gmsh.model.getEntities(2), True)

    # Compute the resulting shear zone surface DimTags by disregarding DimTags of dim 2 in out_frac
    # These are parts of the fractures "sticking out" of the fracture zone.
    out_frac_2d = [o for o in out_frac if o[0] == 2]
    frac_map = {}
    volume_dimtags = []
    for entity_name, tags in zip(["Volume"] + frac_names, map_frac):
        frac_tags = [t for t in tags if t not in out_frac_2d]
        if entity_name == "Volume":
            volume_dimtags = frac_tags
        else:
            frac_map[entity_name] = frac_tags
        if verbose:
            logger.info(
                f"Entity {entity_name} now has {len(frac_tags)} tags: {frac_tags}"
            )

    # The boundary between the outer and inner (fragmented) volumes needs to be re-aligned.
    # To do this, we fragment the outer volume with the inner volumes.
    ent3d = gmsh.model.getEntities(3)
    outer_volume = [ent3d[0]]
    inner_volumes = ent3d[1:]
    # fmt: off
    kernel.fragment(
        outer_volume,
        inner_volumes,
    )
    # fmt: on
    kernel.synchronize()

    # --- Add Physical Groups ------------------------------------------------------------------------------------------

    # Constrain the 3D volumes
    tags3dim = [t[1] for t in gmsh.model.getEntities(3)]
    volume_group = gmsh.model.addPhysicalGroup(3, tags3dim)
    gmsh.model.setPhysicalName(3, volume_group, "DOMAIN")

    # Constrain the 2D fractures
    all_frac_surface_dimtags = []
    fracnum_fracname: Dict[int, str] = {}

    for i, fname in enumerate(frac_names):
        fdimtags = frac_map[fname]
        all_frac_surface_dimtags.extend(fdimtags)

        # -- Method A: Set one fracture element per fracture
        ftags = [t[1] for t in fdimtags]
        fgroup = gmsh.model.addPhysicalGroup(2, ftags)
        gmsh.model.setPhysicalName(2, fgroup, f"FRACTURE_{i}")
        fracnum_fracname[i] = fname

    # Constrain the 1D fracture intersections
    all_frac_lines = gmsh.model.getBoundary(
        all_frac_surface_dimtags, combined=False, oriented=False
    )
    count = Counter(all_frac_lines)
    intersection_dimtags = [i for i in count if count[i] > 1]
    intersection_tags = [t[1] for t in intersection_dimtags]

    if len(intersection_dimtags) > 0:
        for index, itag in enumerate(intersection_tags):
            igroup = gmsh.model.addPhysicalGroup(1, [itag])
            gmsh.model.setPhysicalName(1, igroup, f"FRACTURE_LINE_{index}")

        if verbose:
            logger.info(f"intersection dimtags {intersection_dimtags}")

    # Set a fracture zone physical group for convenience (to view the fracture zone only in gmsh)
    fraczone_tags = [v[1] for v in volume_dimtags]
    fzgroup = gmsh.model.addPhysicalGroup(3, fraczone_tags)
    gmsh.model.setPhysicalName(3, fzgroup, "FRACTURED_ZONE")

    # --- Set mesh size ------------------------------------------------------------------------------------------------

    # Define small mesh elements within the fractured zone and larger outside.
    field = gmsh.model.mesh.field
    box = field.add("Box")
    field.setNumber(box, "VIn", lcin)
    field.setNumber(box, "VOut", lcout)
    field.setNumber(box, "XMin", xmin - dx / 10)
    field.setNumber(box, "XMax", xmax + dx / 10)
    field.setNumber(box, "YMin", ymin - dy / 10)
    field.setNumber(box, "YMax", ymax + dy / 10)
    field.setNumber(box, "ZMin", zmin - dz / 10)
    field.setNumber(box, "ZMax", zmax + dz / 10)

    # Add other field types (if needed, see: t10.py in gmsh tutorial)...
    # Define smaller mesh elements near intersections
    # 1. Define a distance field on the intersection curves
    idist = field.add("Distance")
    field.setNumber(idist, "NNodesByEdge", 100)
    field.setNumbers(idist, "EdgesList", intersection_tags)
    # 2. Define a Threshold field
    # LcMax -                         /------------------
    #                               /
    #                             /
    #                           /
    # LcMin -o----------------/
    #        |                |       |
    #      Point           DistMin DistMax
    ithresh = field.add("Threshold")
    field.setNumber(ithresh, "IField", idist)
    field.setNumber(ithresh, "LcMin", lcin / 2)
    field.setNumber(ithresh, "LcMax", lcout)
    field.setNumber(ithresh, "DistMin", 1 / length_scale)
    field.setNumber(ithresh, "DistMax", 2 / length_scale)

    # Define smaller mesh elements near fractures
    # 1. Define a distance field on the fracture surfaces
    fdist = field.add("Distance")
    all_frac_surface_tags = [t[1] for t in all_frac_surface_dimtags]
    field.setNumbers(fdist, "FacesList", all_frac_surface_tags)
    # 2. Define a Threshold field
    fthresh = field.add("Threshold")
    field.setNumber(fthresh, "IField", fdist)
    field.setNumber(fthresh, "LcMin", lcin / 2.5)
    field.setNumber(fthresh, "LcMax", lcout)
    field.setNumber(fthresh, "DistMin", 1 / length_scale)
    field.setNumber(fthresh, "DistMax", 2 / length_scale)

    # Use the minimum of the fields as the background mesh field
    fmin = field.add("Min")
    field.setNumbers(fmin, "FieldsList", [box, ithresh, fthresh])

    field.setAsBackgroundMesh(fmin)

    # # When the element size is fully specified by a background mesh (as it is in
    # # this example), it is thus often desirable to set
    # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    # # This will prevent over-refinement due to small mesh sizes on the boundary.

    kernel.synchronize()
    gmsh.model.mesh.generate(3)
    if n_optimize_netgen > 0:
        gmsh.model.mesh.optimize("Netgen", niter=n_optimize_netgen)

    if tempdir is True:
        # If path is not provided, store the msh file in a temporary directory.
        with tempfile.TemporaryDirectory() as fp:
            mshfile = str(fp) + "/gmsh_frac_file.msh"
            gmsh.write(mshfile)
            gb: pp.GridBucket = dfm_from_gmsh(mshfile, 3)
    else:
        gmsh.write(str(mshfile))
        gb: pp.GridBucket = dfm_from_gmsh(str(mshfile), 3)

    if run_gmsh_gui:
        gmsh.fltk.run()

    if use_logger:
        gmsh.logger.stop()
    gmsh.finalize()

    # Tag each grid with the shear zone name
    fracs = gb.grids_of_dimension(2)
    gb.add_node_props(["name"])
    for f in fracs:
        fname: str = fracnum_fracname[f.frac_num]
        gb.set_node_prop(f, key="name", val=fname)

    pp.contact_conditions.set_projections(gb)

    return gb
