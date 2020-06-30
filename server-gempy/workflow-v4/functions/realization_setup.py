import gempy as gp
import numpy as np


def set_congiguration(geo_model, geo_model_extent, section):
    """Sets geo_model extent, resolution and section."""

    # Set extent
    # Fix for Bug  # Replace with geo_model.set_extent(extent=extent)
    gp.init_data(
        geo_model,
        extent=geo_model_extent,
        resolution=[10, 10, 10]
    )

    # Set grids
    section_dict = {'section': (
        section['p1'],
        section['p2'],
        section['resolution']
    )}
    geo_model.set_section_grid(section_dict=section_dict)


def update_series(geo_model, series_df):
    """Updates series of the geo-model to the one stored in data.

    Deletes currently existing series in geo_model, sets series passed in
    series_df and sets faults.
    Note: TO_DELETE added as empty series throw an error;

    Args:
        geo_model = The geo_model
        series_df: DataFrame = containing series data
    """

    # remove old state  # gempy does not allow emtpy sereies
    old_series = geo_model.series.df.index.to_list()
    geo_model.add_series(series_list=['TO_DELETE'])

    try:

        geo_model.delete_series(old_series)

    except:

        pass

    # set new state
    series_df.sort_index()
    new_series = series_df['name'].to_list()
    geo_model.add_series(new_series)

    # HOTFIX
    try:

        geo_model.delete_series(['TO_DELETE'])
        
    except:

        pass

    print('HOTFIX in update_series()')


def update_surfaces(geo_model, surfaces_df):
    """Updates surfaces of the geo-model to the one stored in data.

    Deletes currently existing surfaces in geo_model and sets surfaces passed
    in surfaces_df.
    Loops over surfaces to map them to series.

    Args:
        geo_model = The geo_model
        series_df: DataFrame = containing surface data
    """

    # remove old state
    old_surfaces = geo_model.surfaces.df['surface'].to_list()
    try:
        geo_model.delete_surfaces(old_surfaces)
    except:
        print('HOTFIX in update_surfaces()')

    # set new state
    surfaces_df.sort_index()
    new_surfaces = surfaces_df['name'].to_list()
    geo_model.add_surfaces(new_surfaces)


def creat_mapping_object(surfaces_df, series_df):

    surfaces_df.sort_values(by=['order_surface'])
    mapping_object = {}
    for index, row in series_df.iterrows():

        series_name = row['name']
        categories = surfaces_df[surfaces_df['serie'] == series_name]['name'].astype('category')
        mapping_object[series_name] = categories

    return mapping_object


def setup_realization(
        geo_model,
        geo_model_extent,
        section,
        series_df,
        surfaces_df,
        surface_points_original_df,
        orientations_original_df
):
    
    set_congiguration(geo_model, geo_model_extent, section)
    
    update_series(geo_model, series_df)
    
    update_surfaces(geo_model, surfaces_df)
    
    geo_model.set_surface_points(surface_points_original_df, update_surfaces=False)
    geo_model.set_orientations(orientations_original_df, update_surfaces=False)
    
    mapping_object = creat_mapping_object(surfaces_df, series_df)    
    gp.map_series_to_surfaces(
        geo_model=geo_model,
        mapping_object=mapping_object
    )
    
    if np.any(series_df['isfault']):

        fault_series = series_df[series_df['isfault']]['name']
        geo_model.set_is_fault(fault_series)    
        
    geo_model.update_to_interpolator()