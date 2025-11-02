import xarray as xr
from xarray.ufuncs import angle
import numpy as np


def berry_curvature(eigve: xr.DataArray, dims = ["kx","ky"], compdim = 'component') -> xr.DataArray:
    """Compute the Berry curvature for all eigenvector maps stored in 'eigve', using the 4-point formula

    Args:
        eigve (xr.DataArray): A xarray DataArray containing the eigenvectors maps and some eventual additional dimensions. 
        The array must at least be 3D with one dimension for the eigenvector component and two dimensions for the reciprocal space.
        dims (list, optional): The reciprocal space dimensions. Defaults to ["kx","ky"].
        compdim (str, optional): Name of the eigenvector component dimension. Default to 'component'.
    Returns:
        xr.DataArray: The computed Berry curvature
    """
    
    if compdim not in eigve.dims:
        raise ValueError(f"{compdim} not a dimension of eigve")
    for dim in dims:
        if dim not in eigve.dims:
            raise ValueError(f"{dim} not a dimension of eigve")
        
    eigve1 = eigve.shift({dims[0]:1}, fill_value=0)
    eigve3 = eigve.shift({dims[1]:1}, fill_value=0)
    eigve2 = eigve.shift({dims[0]:1,dims[1]:1}, fill_value=0)
    
    p1 = xr.dot(eigve.conj(), eigve1, dims = compdim)
    p2 = xr.dot(eigve1.conj(), eigve2, dims = compdim)
    p3 = xr.dot(eigve2.conj(), eigve3, dims = compdim)
    p4 = xr.dot(eigve3.conj(), eigve, dims = compdim)
    
    return -angle(p1*p2*p3*p4)
