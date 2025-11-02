import numpy as np
import xarray as xr
from typing import Union
from copy import deepcopy
from xarray_einstats.linalg import eigh
from types import NoneType
from .geometry import stringtoarray, arraytostring

def angle(data:xr.DataArray)->xr.DataArray:
    return xr.apply_ufunc(np.angle, data)

def conjugate(data:xr.DataArray)->xr.DataArray:
    return xr.apply_ufunc(np.conjugate, data)

