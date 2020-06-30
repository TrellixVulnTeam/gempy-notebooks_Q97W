import numpy as np  # type: ignore
import gempy as gp  # type: ignore
from operator import itemgetter  # type: ignore


def compute_boolean_matrix_for_section_surface_top(geo_model=gp.Model):
    """ Compute points in the section grid that mark the transtion of one
    surface to another.

    Args:
        geo_model = geo_model instance
        surface_index = index of wanted surface

    Return:
        np.array() = Boolen matrix represention scalar-field transitions of
            surface-i top;
    """

    # get faults if any
    faults = list(geo_model.faults.df[geo_model.faults.df['isFault'] == True].index)

    # get start and end of section in grid scalar vals array
    arr_len_0, arr_len_n = geo_model.grid.sections.get_section_args('section')

    # CAUTION: if more section present they have to be indexexed accrodingly;
    # get shape of section  # 1st and only one here as only one section present.
    section_shape = geo_model.grid.sections.resolution[0]
    # extract section scalar values from solutions.sections# [series,serie_pos_0:serie_pos_n]
    section_scalar_field_values = geo_model.solutions.sections[1][:,arr_len_0:arr_len_n]
    # get extent
    extent = [
        0,
        geo_model.grid.sections.dist[0][0],
        geo_model.grid.regular_grid.extent[4],
        geo_model.grid.regular_grid.extent[5]
    ]

    # number scalar field blocks
    n_scalar_field_blocks = section_scalar_field_values.shape[0]
    # creat a dictionary to assemble all scalat field boolen matrices shifts
    # extract transition towards current level
    matrix_shifts_results = {}
    for i in range(n_scalar_field_blocks):

        # scalarfield values of scalarfield_block-i
        block = section_scalar_field_values[i,:]
        # ??? level 
        level = geo_model.solutions.scalar_field_at_surface_points[i][np.where(
                geo_model.solutions.scalar_field_at_surface_points[i] != 0)]
        # ??? calulcate scalarfeild levels
        levels = np.insert(level, 0, block.max())
        # extract and reshape scalar field values
        scalar_field = block.reshape(section_shape).T
        # loop over levels to extract tops present within current scalar field
        for ii in range(len(levels)):

            # B: boolean matrix of scalar field valeus > current level value
            B = scalar_field > levels[ii]
            # matrix shifting to get transition to current level
            B_0 = B[1:, :]
            B_1 = B[:-1, :]
            B01 = B_0 ^ B_1
            # append to results dictionary
            matrix_shifts_results[f'{i}-{ii}'] = B01

    return matrix_shifts_results


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

    # get xvals and yvals
    xvals = xy_coord_on_line_p0_p1[0]
    yvals = xy_coord_on_line_p0_p1[1]

    # stretching whole extent
    zvals = np.linspace(z_min, z_max, resolution[1])

    # meshgrids to get grid coordinates
    X, Z = np.meshgrid(xvals, zvals)
    Y, Z = np.meshgrid(yvals, zvals)

    return np.stack((X, Y, Z))