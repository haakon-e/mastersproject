from GTS.isc_modelling.cm_biot_ex1.data import Data

path = "home/haakon/mastersproject/src/mastersproject/GTS/isc_modelling/cm_biot_ex1/results"


# Define mesh sizes
mesh_size = 100
mesh_args = {
    "mesh_size_frac": mesh_size,
    "mesh_size_min": 0.1 * mesh_size,
    "mesh_size_bound": 6 * mesh_size,
}
setup = Data(mesh_args, folder_name=path)
