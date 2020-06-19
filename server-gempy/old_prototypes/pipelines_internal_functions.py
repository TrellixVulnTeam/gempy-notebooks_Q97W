import copy

import gempy as gp
import numpy as np
import scipy.stats as ss


def update_series(geo_model, series_df):
    """Updates series of the geo-model to the one stored in data."""

    series_old = list(geo_model.series.df.to_dict()['order_series'].keys())

    # add new series
    for index, row in series_df.iterrows():

        serie_name = row['name']
        if serie_name not in series_old:

            geo_model.add_series(series_list=[serie_name])

    # remove obsolete series
    for serie in series_old:

        if serie not in series_df['name'].to_list():

            geo_model.delete_series(serie)


def update_faults_relations(geo_model, series_df):
    """Sets fault relations."""

    for index, row in series_df.iterrows():

        serie_name = row['name']
        serie_isfault = row['isfault']
        if serie_isfault:

            geo_model.set_is_fault([serie_name])


def update_surfaces(geo_model, surfaces_df):
    """Updates surfaces of the geo-model to the one stored in data."""

    surfaces_old = geo_model.surfaces.df['surface'].to_list()

    # add and update surfaces
    for index, row in surfaces_df.iterrows():

        surface_name = row['name']
        surface_serie = row['serie']
        if surface_name not in surfaces_old:

            geo_model.add_surfaces(surface_list=[surface_name])
            gp.map_series_to_surfaces(
                geo_model,
                {surface_serie: surface_name}
            )

        else:

            gp.map_series_to_surfaces(
                geo_model,
                {surface_serie: surface_name}
            )

    # remove obsolete surfaces
    for surface in surfaces_old:

        if surface not in surfaces_df['name'].to_list():

            geo_model.delete_surfaces(surface)


def run_realizations(geo_model, n_realizations, surface_points_original_df, orientations_df, series_df):

    # Copy geological input data to manipulate per realization.
    surface_points_copy = copy.deepcopy(surface_points_original_df)

    # Storage for calucalted ralizations
    list_section_data = []

    # TODO: Move Topological Realtaions updates to here

    # Calculate realizations
    for i in range(n_realizations):

        # manipulate surface_points_copy in place
        surface_points_copy['X'] = ss.norm.rvs(
            loc=surface_points_original_df['X'].values,
            scale=surface_points_original_df['param1'].values)
        surface_points_copy['Y'] = ss.norm.rvs(
            loc=surface_points_original_df['Y'].values,
            scale=surface_points_original_df['param1'].values)
        surface_points_copy['Z'] = ss.norm.rvs(
            loc=surface_points_original_df['Z'].values,
            scale=surface_points_original_df['param1'].values)

        # Data to model
        # TODO: Replace with function
        gp.init_data(
            geo_model,
            extent=[0, 2000, 0, 2000, 0, 2000],
            resolution=[5, 5, 5],
            surface_points_df=surface_points_copy,
            orientations_df=orientations_df,
            update_surfaces=False
        )

        # Set fault realtions
        for index, row in series_df.iterrows():

            serie_name = row['name']
            serie_isfault = row['isfault']
            if serie_isfault:

                geo_model.set_is_fault([serie_name])

        # update to interpolator
        geo_model.update_to_interpolator()

        # Set section grid  # Only one => client canvas
        # TODO: Deactivate regular sectio  # Best case on init of geo_model
        geo_model.set_section_grid(section_dict=section_dict)

        # Compute solution
        # TODO: Fix bug!
        # till here: until 90.1 ms for 1 realizations
        # 213 m with 2x gp.compute_model()
        solution = gp.compute_model(model=geo_model)
        solution = gp.compute_model(model=geo_model)

        # collect extracted section data
        list_section_data.append(geo_model
                                 .solutions
                                 .sections[0][0]
                                 .reshape(section_dict['section'][2])
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
        section_dict
):

    count_array = np.empty((
        section_dict['section'][2][0],
        section_dict['section'][2][1],
        len(lithology_ids)))

    for index, lithology in enumerate(lithology_ids):

        count_array[:, :, index] = np.sum((
            section_data_stack == lithology).astype(int), axis=2)

    return count_array


def calculate_information_entropy(count_array, n_realizations):

    # Calculate information entropy
    return ss.entropy(probability_array, axis=2)
