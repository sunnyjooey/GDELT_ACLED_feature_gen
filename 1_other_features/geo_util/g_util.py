import pandas as pd
import rasterio
from rasterio.plot import show, show_hist
import fiona
import geopandas as gpd
from rasterstats import zonal_stats
import matplotlib.pyplot as plt


def test_compatibility(shapefile_path, raster_path):
    """
    Function used to test compatibility of two files in terms of metadata

    Inputs:
        shapefile_path: the path of the shape file
        aster_path: the path of the raster file
    Outputs:
        Print statements specifying the result
    """
    
    print('***Results of Compatiblity Check:*** ')
    print('\n')
    
    with rasterio.open(raster_path) as raster:
        raster_crs = raster.crs
        raster_bounds = raster.bounds
        raster_res = raster.res
        raster_nodata = raster.nodata
        raster_dtype = raster.dtypes[0]
        
    with fiona.open(shapefile_path) as shapefile:
        shapefile_crs = shapefile.crs
        shapefile_schema = shapefile.schema
    
    # 1. Coordinate System / Projection
    if raster_crs == shapefile_crs:
        print("Coordinate systems are compatible!")
    else:
        print("Coordinate systems are NOT compatible!")
    print('\n')

    # 2. Spatial Extent
    gdf = gpd.read_file(shapefile_path)
    overlaps = True

    if gdf.total_bounds[0] < raster_bounds.left:
        print("The left boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if gdf.total_bounds[1] < raster_bounds.bottom:
        print("The bottom boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if gdf.total_bounds[2] > raster_bounds.right:
        print("The right boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if gdf.total_bounds[3] > raster_bounds.top:
        print("The top boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if overlaps:
        print("Spatial extents are compatible!")
    else:
        print("Spatial extents do not perfectly overlap!")
    print('\n')
        
    # 3. Resolution (for raster)
    print(f"Raster resolution: {raster_res}")
    print('\n')
    
    # 4. NoData Value (for raster)
    print(f"NoData value for the raster: {raster_nodata}")
    print('\n')
    
    # 5. Attribute Data (for vector)
    print("Shapefile attributes:")
    print(shapefile_schema)
    print('\n')
    
    # 6. Data Types
    print(f"Raster data type: {raster_dtype}")  
    print(f"Vector data types: {shapefile_schema['properties']}")
    print('\n')
    

def geo_plot(shapefile_path, raster_path, hist=False):
    """
    Give an initial plot of the combination of the shape file and the ratser file
    
    Inputs:
        shapefile_path: the path of the shape file
        aster_path: the path of the raster file
        hist(bool): check if a histogram is needed
    Outputs:
        A plot
    """
    
    shape_gdf = gpd.read_file(shapefile_path)
    raster_file = rasterio.open(raster_path, mode='r')
    
    if hist:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
        show(raster_file, ax=ax1, title="Population Density 2020")
        shape_gdf.plot(ax=ax1, facecolor="None", edgecolor="yellow")
        show_hist(raster_file, ax=ax2, title="Histogram of Population Density")
    else:
        fig, ax1 = plt.subplots(1,1)
        show(raster_file, ax=ax1, title="Population Density 2020")
        shape_gdf.plot(ax=ax1, facecolor="None", edgecolor="yellow")
    
    plt.show()


def merge_geo(shapefile_path, raster_path):
    """
    Merge two datasets by calculating the zonal statistics
    
    Inputs:
        shapefile_path: the path of the shape file
        aster_path: the path of the raster file
    Outputs:
        stats: a dict of lists containing relevant stats
    """

    # Calculate the zonal stats
    stats = zonal_stats(shapefile_path, raster_path)
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    # Convert the list of dictionaries to a DataFrame
    stats_df = pd.DataFrame(stats)
    # Join the zonal statistics DataFrame with the original GeoDataFrame
    gdf_stats = gdf.join(stats_df)
    
    return gdf_stats