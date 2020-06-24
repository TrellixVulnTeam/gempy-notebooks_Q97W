import copy

import gempy as gp
import numpy as np
import scipy.stats as ss


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

    # set faults
    if np.any(series_df['isfault']):

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


def manipulate_surface_points_inplace(surface_points_copy, surface_points_original_df):
    """Manipulates the surface_points_copy dataframe.
    
        Samples X, Y, Z values form the original DataFrame and thier
        respective distribution types and parameters.\
        Potential update:
            - Sampling parameter per axis i.e. param1_x, param1_y, ...
            - Diffenrent sampling types i.e. normal, uniformal, ...
    
        Args:
            surface_points_copy: DataFrame = copy of the original geological
                input data surface-points DataFrame.
            surface_points_original_df: DataFrame = original geological input data
                surface-points DataFrame.
    """
    
    surface_points_copy['X'] = ss.norm.rvs(
        loc=surface_points_original_df['X'].values,
        scale=surface_points_original_df['param1'].values)
    surface_points_copy['Y'] = ss.norm.rvs(
        loc=surface_points_original_df['Y'].values,
        scale=surface_points_original_df['param1'].values)
    surface_points_copy['Z'] = ss.norm.rvs(
        loc=surface_points_original_df['Z'].values,
        scale=surface_points_original_df['param1'].values)

def run_realizations(
    geo_model,
    n_realizations,
    surface_points_original_df,
    orientations_original_df,
    section
):
    """Runs x ralizations"""

    # Copy geological input data to manipulate per realization.
    surface_points_copy = copy.deepcopy(surface_points_original_df)

    # Storage for calucalted ralizations
    list_section_data = []

    # Calculate realizations
    for i in range(n_realizations):

        print(f'Realization: {i}')
        
        # manipulate surface_points_copy in place
        manipulate_surface_points_inplace(
            surface_points_copy=surface_points_copy,
            surface_points_original_df=surface_points_original_df)
        
        # Set manipulated surface points
        geo_model.set_surface_points(surface_points_copy, update_surfaces=False)
        # geo_model.set_orientations(orientations_original_df, update_surfaces=False)

        # update to interpolator
        geo_model.update_to_interpolator()

        # Compute solution
        # TODO: Fix bug!
        # till here: until 90.1 ms for 1 realizations
        # 213 m with 2x gp.compute_model()
        try:
            solution = gp.compute_model(model=geo_model)
            solution = gp.compute_model(model=geo_model)
            # gp.plot.plot_section_by_name(geo_model, 'section')
        except ValueError as err:
            print('ValueError')
            # Append last working realization
            list_section_data.append(geo_model
                                     .solutions
                                     .sections[0][0]
                                     .reshape(section['resolution'])
                                     )

            
        # collect extracted section data
        list_section_data.append(geo_model
                                 .solutions
                                 .sections[0][0]
                                 .reshape(section['resolution'])
                                 )

    return list_section_data


def process_list_section_data(list_section_data):

    # Process results Stack results
    section_data_stack = np.round(np.dstack(list_section_data))

    # Get lithologies in stack
    lithology_ids = np.unique(section_data_stack)

    return section_data_stack, lithology_ids


def count_lithology_occurrences_over_realizations(
        section_data_stack,
        lithology_ids,
        section
):
    
    count_array = np.empty((
        section['resolution'][0],
        section['resolution'][1],
        len(lithology_ids)))

    for index, lithology in enumerate(lithology_ids):

        count_array[:, :, index] = np.sum((
            section_data_stack == lithology).astype(int), axis=2)

    return count_array


def calculate_information_entropy(count_array, n_realizations):

    # Calculate information entropy
    probability_array = count_array / n_realizations
    return ss.entropy(probability_array, axis=2)
