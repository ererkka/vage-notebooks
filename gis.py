# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:21:32 2018

@author: ererkka
"""

import xarray as xa
import netCDF4  # Ensure netCDF library gets loaded before rasterio
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import rasterio.mask
import numpy as np
from numpy import pi, sin, deg2rad
import dask.array as da


NODATA = -9999  # No data value for rasterio
REF_EARTH_RADIUS =  6367470.0  # Reference spherical Earth radius (m)


#%%
def cell_area(lat, lon, xsize, ysize, radius=REF_EARTH_RADIUS):
    """Calculate earth surface area of grid cell at (lat, lon)
    
    Parameters
    ----------
    lat, lon: float
        Latitude and longitude (deg) coordinates of the (centre) of the grid cell
    xsize, ysize, : float
        Longitudal and latitudal size of grid cells (deg)
        
    Returns
    -------
    cell size: float
    """
    
    lon1 = deg2rad(lon - xsize / 2)
    lon2 = deg2rad(lon + xsize / 2)
    
    lat1 = deg2rad(lat - ysize / 2)
    lat2 = deg2rad(lat + ysize / 2)
    
    # From: http://mathforum.org/library/drmath/view/63767.html
    return radius**2 * abs(sin(lat2) - sin(lat1)) * abs(lon2 - lon1)


#%%
def add_zone_dim(data, zones, shapes, crs, xsize, ysize, dim='zone'):
    """Add zone dimension to an xarray DataArray based on polygon shapes
    
    Parameters
    ----------
    data: xarray.DataArray
    zones: list 
        Names of zones
    shapes: list
        Shapes of zones. Each shape must be GeoJSON-like dict
    crs: str
        Coorindate reference system in PROJ4 format
    xsize, ysize: float
        Size of pixel in degrees longitude or latitude
    dim: str, optional
        Name of zone dimension (default: 'zone')
    """
    expanded = data.expand_dims(dim, axis=0)
    
    transform = from_origin(west=data.lon.values[0], north=data.lat.values[-1],
                            xsize=xsize, ysize=ysize)
    
    def block_func(array, shapes):
        arr = array[0]
        memfile = MemoryFile()    
        with memfile.open(driver='MEM', 
                          width=arr.shape[2], height=arr.shape[1],
                          count=arr.shape[0], dtype=str(arr.dtype),
                          crs=crs, transform=transform, 
                          nodata=NODATA) as raster:
    
            raster.write(arr, indexes=list(range(1, arr.shape[0] + 1)))
    
            arrays = list()
            for shp in shapes:
                masked_arr, _ = rasterio.mask.mask(raster, [shp])
                masked_arr[masked_arr == NODATA] = np.nan
                arrays.append(masked_arr)
    
        return np.stack(arrays, axis=0)
    
    if data.chunks is not None:
        mapped = da.map_blocks(block_func, expanded.data, shapes, 
                               dtype=data.dtype, 
                               chunks=((len(shapes),),) + data.chunks)
    else:
        mapped = block_func(expanded.values, shapes)
    arr = xa.DataArray(mapped, dims=expanded.dims, coords=expanded.coords)
    arr[dim] = zones
    return arr

