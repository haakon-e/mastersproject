import math

import gmsh


def volume_with_fracture():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("volume-with-fracture")
    gmsh.logger.start()
    kernel = gmsh.model.occ

    # Add outer box
    dx = dy = dz = 1
    outer_box = kernel.addBox(0, 0, 0, dx, dy, dz)

    # Add inner box
    cx = cy = cz = dx / 2  # Center  # Set to dx/2 in production
    sx = sy = sz = 0.2  # Radius
    corner = (cx - sx, cy - sy, cz - sz)
    size = (2 * sx, 2 * sy, 2 * sz)
    inner_box = kernel.addBox(*corner, *size)
    kernel.synchronize()

    # Add fracture
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, inner_box)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    angle = -math.pi / 6
    lx = dx / math.cos(angle)

    # Through-going fracture
    frac = kernel.addRectangle(xmin, ymin, zmin, lx, dy)
    # Rotate
    f = [(2, frac)]
    kernel.rotate([f[0]], xmin, ymin, zmin, 0, 1, 0, angle)

    # Insert the fracture to the domain
    f_out, f_m = kernel.fragment([(3, inner_box)], [f[0]])
    kernel.synchronize()

    # Partition domain
    box_dim = 3
    out, m = kernel.fragment([(box_dim, outer_box)], f_out,)
    kernel.synchronize()

    # Mesh the domain
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)

    gmsh.model.mesh.generate(3)

    gmsh.fltk.run()
    gmsh.finalize()


def domain_partition():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("fractured-volume-by-fragment")
    gmsh.logger.start()
    kernel = gmsh.model.occ

    # Copy inner box from 'volume-empty-center.py'
    cx = cy = cz = 0.2  # Center  # Set to d/2 in production
    sx = sy = sz = 0.2  # Radius
    corner = (cx - sx, cy - sy, cz - sz)
    size = (2 * sx, 2 * sy, 2 * sz)
    inner_box = kernel.addBox(*corner, *size)

    kernel.synchronize()

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, inner_box)
    dx = xmax - xmin
    dy = ymax - ymin
    # dz = zmax - zmin

    angle = -math.pi / 6
    lx = dx / math.cos(angle)

    # Throughgoing fracture
    frac = kernel.addRectangle(xmin, ymin, zmin, lx, dy)
    # Rotate
    f = [(2, frac)]
    kernel.rotate([f[0]], xmin, ymin, zmin, 0, 1, 0, angle)

    box_dim = 3
    frac_dim = 2
    # Insert the fracture to the domain
    kernel.fragment([(box_dim, inner_box)], [(frac_dim, frac)])

    kernel.synchronize()
    gmsh.fltk.run()
    gmsh.finalize()


# if __name__ == "__main__":
#     volume_with_fracture()
