import porepy as pp
import numpy as np
from porepy.models.contact_mechanics_model import ContactMechanics


class ContactMechanicsEx(ContactMechanics):
    """ Implementation of ContactMechanics with a workable example"""

    def __init__(self):
        """ Overwrite the __init__ class"""
        super().__init__()

        # Time
