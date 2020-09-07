import numpy as np


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

    return True