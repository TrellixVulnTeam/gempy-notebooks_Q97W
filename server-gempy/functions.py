import gempy as gp


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

    Loops over series and appends missing ones and removes obsolet ones.

    Args:
        geo_model = The geo_model
        series_df: DataFrame = containing series data
    """

    # remove old state
    old_series = geo_model.series.df.index.to_list()
    geo_model.add_series(series_list=['TO_DELETE'])
    geo_model.delete_series(old_series)

    # set new state
    series_df.sort_index()
    geo_model.add_series(series_list=series_df['name'].to_list())
    geo_model.delete_series(['TO_DELETE'])
    geo_model.set_is_fault(series_df[series_df['isfault']]['name'])
