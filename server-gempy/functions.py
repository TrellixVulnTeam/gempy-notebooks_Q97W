import gempy as gp
import numpy as np


def sanity_check():

    print('Hello functions.py')


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

    # remove old state
    old_series = geo_model.series.df.index.to_list()
    geo_model.add_series(series_list=['TO_DELETE'])
    
    try:    
        geo_model.delete_series(old_series)
    except:
        pass

    # set new state
    series_df.sort_index()
    new_series = series_list = series_df['name'].to_list()
    geo_model.add_series(new_series)
    # HOTFIX
    try:    
        geo_model.delete_series(['TO_DELETE'])
    except:
        pass    

    # set faults
    fault_series = series_df[series_df['isfault']]['name']
    geo_model.set_is_fault(fault_series)

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

    # remove olld state
    old_surfaces = geo_model.surfaces.df['surface'].to_list()
    geo_model.delete_surfaces(old_surfaces)

    # set new state
    surfaces_df.sort_index()
    new_surfaces = surfaces_df['name'].to_list()
    geo_model.add_surfaces(new_surfaces)

    # map to series
    for index, row in surfaces_df.iterrows():
        
        surface = row['name']
        serie = row['serie']
        gp.map_series_to_surfaces(
            geo_model,
            {serie: surface}
        )


def check_setup_single_realization(geo_model):

    print('Run realizations setup checks until stable workflow.')
    
    # check if surface_points are within geo-model-extent
    current_extent = geo_model.grid.regular_grid.extent
    if not (
        current_extent[0] <= np.min(geo_model.surface_points.df['X'].values) and
        current_extent[1] >= np.max(geo_model.surface_points.df['X'].values) and
        current_extent[2] <= np.min(geo_model.surface_points.df['Y'].values) and
        current_extent[3] >= np.max(geo_model.surface_points.df['Y'].values) and
        current_extent[4] <= np.min(geo_model.surface_points.df['Z'].values) and
        current_extent[5] >= np.max(geo_model.surface_points.df['Z'].values)
    ):
        raise ValueError(f'Some surface-poins are not within geo-model-extent-bounds')
        
    # check if orientations are within geo-model-extent
    if not (
        current_extent[0] <= np.min(geo_model.orientations.df['X'].values) and
        current_extent[1] >= np.max(geo_model.orientations.df['X'].values) and
        current_extent[2] <= np.min(geo_model.orientations.df['Y'].values) and
        current_extent[3] >= np.max(geo_model.orientations.df['Y'].values) and
        current_extent[4] <= np.min(geo_model.orientations.df['Z'].values) and
        current_extent[5] >= np.max(geo_model.orientations.df['Z'].values)
    ):
        raise ValueError(f'Some orientations are not within geo-model-extent-bounds')
    
    # check if at least one orientaion per series
    orientation_series = geo_model.orientations.df['series'].unique()
    geo_model_series = list(geo_model.series.df.index)
    if not all([serie in geo_model_series for serie in orientation_series]):
        
        raise ValueError(f'Some series have no orientaion')
    
    
    # check if at least two surface-points per surface
    surfaces_surface_points = geo_model.surface_points.df
    surfaces_geo_model = list(geo_model.surfaces.df['surface'])
    for surface in surfaces_geo_model:

        if not surface == 'basement':
            len_df = len(surfaces_surface_points[surfaces_surface_points['surface'] == surface])
            if len_df < 2:

                raise ValueError(f'Each surface needs at least 2 surface points.')