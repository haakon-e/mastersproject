import porepy as pp
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot

import logging

logging.basicConfig(level=logging.DEBUG)


class Data(ContactMechanicsBiot):
    def __init__(self, mesh_args, folder_name):
        params = {"folder_name": folder_name}
        super().__init__(self, params)
