import numpy as np
import pandas as pd
import matplotlib.pylab as plt


def csv_to_feather(csv_source_str: str = None, feather_dest_str: str = None, index_col: int = 0, sep: str = ';') -> pd.DataFrame:
    """Convert CSV source file to a feather, and return a pandas dataframe.

    Args:
        csv_source_str (str, optional): CSV file path as a string. Defaults to None.
        feather_dest_str (str, optional): Feather destination file path. Defaults to None.
        index_col (int, optional): index column. Defaults to 0.

    Returns:
        pd.DataFrame: returned pandas dataframe object
    """
    if csv_source_str is None or feather_dest_str is None:
        return None
    try:
        D = pd.read_csv(csv_source_str, sep=sep, index_col=index_col)
        D.to_feather(feather_dest_str)
        featherD = pd.read_feather(feather_dest_str)
    except Exception as e:
        print(f'Error transforming csv into feather: {e}')
    return featherD
