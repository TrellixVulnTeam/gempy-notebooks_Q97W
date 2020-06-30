import copy
from operator import itemgetter  # type: ignore

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


def compute_boolean_matrix_for_section_surface_top(
    geo_model=gp.Model,
    surface_index=int
):
    """ Compute points in the section grid that mark the transtion of one
    surface to another.

    Args:
        geo_model = geo_model instance
        surface_index = index of wanted surface

    Return:
        np.array() = Boolen matrix represention scalar-field transitions of
            surface-i top;
    """

    # get data of current geo_model
    section_shape = geo_model.grid.sections.resolution[0]
    section_scalar_field_values = geo_model.solutions.sections[1]

    # get scalar field level boundaries seperating lithologies
    # marking surface tops from bottom to top
    level = geo_model.solutions.scalar_field_at_surface_points[0]

    # reshape 1D NpArray to original section shape
    section_scalar_field_values_reshaped = section_scalar_field_values[0, :] \
        .reshape(section_shape) \
        .T

    # find scalar field values biggern then level-boundary value
    bigger_level_i = section_scalar_field_values_reshaped > level[surface_index]

    # use matrix shifting along x0axis to find top-surface
    bigger_level_i_0 = bigger_level_i[1:, :]
    bigger_level_i_1 = bigger_level_i[:-1, :]
    bigger_level_i_boundary = bigger_level_i_0 ^ bigger_level_i_1

    return bigger_level_i_boundary   
    

def compute_setction_grid_coordinates(geo_model, extent):

    # extract data
    section_df = geo_model.grid.sections.df.loc['section']
    point_0 = np.array(section_df['start'])
    point_1 = np.array(section_df['stop'])
    resolution = np.array(section_df['resolution'])
    distance = np.array(section_df['dist'])
    z_min, z_max = itemgetter('z_min', 'z_max')(extent)

    # vector pointing from point_0 to point_1
    vector_p0_p1 = point_1 - point_0

    # normalizae vector
    vector_p1_p2_normalizaed = vector_p0_p1 / np.linalg.norm(vector_p0_p1)

    # steps on line between points
    steps = np.linspace(0, distance, resolution[0])

    # calculate xy-coordinates on line between point_0 and point_1
    xy_coord_on_line_p0_p1 = point_0.reshape(2, 1) + vector_p1_p2_normalizaed.reshape(2, 1) * steps.ravel()
    print(xy_coord_on_line_p0_p1.shape)

    # get xvals and yvals
    xvals = xy_coord_on_line_p0_p1[0]
    yvals = xy_coord_on_line_p0_p1[1]

    # stretching whole extent
    zvals = np.linspace(z_min, z_max, resolution[1])

    # meshgrids to get grid coordinates
    X, Z = np.meshgrid(xvals, zvals)
    Y, Z = np.meshgrid(yvals, zvals)

    return np.stack((X, Y, Z))
    
    
def extract_section_coordinates_of_surface(
    geo_model,
    surface_index,
    section_grid_coordinates
) -> np.ndarray:
    """Get coordinates of surface. 
    
    Args:
        geo_model = Server geo model instance.
        surface_index = index on surface on surface stack
        section_grid_coordinates = coordinates of section grid

    Returns
        (3, n_points) of surface coordinates 
    """

    # calculate surface-boolen-matrix "B"
    B = compute_boolean_matrix_for_section_surface_top(
        geo_model=geo_model,
        surface_index=surface_index
    )

    # extract data  # [X|Y|Z][SliceMissingRow#MatrixShifting][BoolenMatrix]
    x_coords = section_grid_coordinates[0][:-1, :][B]
    y_coords = section_grid_coordinates[1][:-1, :][B]
    z_coords = section_grid_coordinates[2][:-1, :][B]

    return np.stack((x_coords, y_coords, z_coords), axis=0)
    

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
