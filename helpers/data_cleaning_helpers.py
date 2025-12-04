import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import typing
from config import MIN_DIMENSION, MAX_DENSITY, MAX_WEIGHT, NUMERIC_COLUMNS_TO_CHANGE_NAMES


def drop_na_rows(df: pd.DataFrame, col_list: typing.List[str] = []) -> pd.DataFrame:
    """Drop rows with no values (NaN). If a column list is given, only check empty values on those columns.

    Args:
        df (pd.DataFrame): dataframe object
        col_list (typing.List[str], optional): column list, for NaN checks. Defaults to [].

    Returns:
        pd.DataFrame: updated dataframe
    """
    try:
        df_rows, df_columns = get_df_shape(df)
        if col_list == []:
            df = df.dropna()
        else:
            df = df.dropna(subset=col_list)
        changed_df_rows, changed_df_columns = get_df_shape(df)
        print_removed_rows_and_columns(df_rows, df_columns, changed_df_rows, changed_df_columns)
    except Exception as e:
        print(f'Error removing na rows: {e}')
    return df


def drop_below_threshold_rows(df: pd.DataFrame, col: str, threshold: int) -> pd.DataFrame:
    """
    Drop all rows where the category in column `col`
    appears less than `threshold` times in the dataframe.

    Example:
        If threshold = 50 and value 'X' appears 30 times → drop all 'X' rows.

    Args:
        df (pd.DataFrame): dataframe object
        col (str, optional): column that is being checked for categories.
        threshold (int, optional): threshold for deleting rows that have a categorical count less than the threshold.

    Returns:
        pd.DataFrame: updated dataframe
    """
    try:
        # Count occurrences of each category
        value_counts = df[col].value_counts()

        # Find categories that meet the threshold
        allowed_categories = value_counts[value_counts >= threshold].index

        # Filter dataframe to keep only these categories
        filtered_df = df[df[col].isin(allowed_categories)].copy()

        # (Optional) print summary
        removed_rows = len(df) - len(filtered_df)
        print(f"Removed {removed_rows} rows where '{col}' count < {threshold}.")

        return filtered_df

    except Exception as e:
        print(f"Error processing column '{col}' with threshold '{threshold}': {e}")
    return df


def drop_str_match_rows(df: pd.DataFrame, col: str = None, string_for_deletion: str = None) -> pd.DataFrame:
    """Drop the rows that have a specific matching string in a specific column.

    Args:
        df (pd.DataFrame): dataframe object
        col (str, optional): the column that holds certain string values. Defaults to None.
        string_for_deletion (str, optional): the string for matching rows that will be deleted. Defaults to None.

    Returns:
        pd.DataFrame: updated dataframe
    """

    if col is None or string_for_deletion is None:
        print(f'Either col {col} or string_for_deletion {string_for_deletion} wasnt given.')
        return df
    try:
        df_rows, df_columns = get_df_shape(df)
        df = df.drop(df[df.get(col) == string_for_deletion].index)
        changed_df_rows, changed_df_columns = get_df_shape(df)
        print_removed_rows_and_columns(df_rows, df_columns, changed_df_rows, changed_df_columns)
    except Exception as e:
        print(f'Unable to remove rows from column {col} according to string match {string_for_deletion}: {e}')
    return df


def get_df_shape(df: pd.DataFrame) -> typing.Tuple[int, int]:
    """Get the shape of dataframe, df.

    Args:
        df (pd.DataFrame): the dataframe object

    Returns:
        Tuple[int, int]: the number of rows and columns returned
    """
    df_rows, df_columns = df.shape
    return df_rows, df_columns


def get_delta_shape(first_df_rows: int, first_df_columns: int, changed_df_rows: int, changed_df_columns: int) -> typing.Tuple[int, int]:
    """Gets the change in the shape of the dataframe. Comparing the before and the after shape.

    Args:
        first_df_rows (int): dataframe row count before changes
        first_df_columns (int): dataframe column count before changes
        changed_df_rows (int): dataframe row count after changes
        changed_df_columns (int): dataframe column count after changes

    Returns:
        Tuple[int, int]: the change in the number of rows, and the change in the number of columns
    """
    delta_df_rows = first_df_rows - changed_df_rows
    delta_df_columns = first_df_columns - changed_df_columns
    return delta_df_rows, delta_df_columns


def print_removed_rows_and_columns(first_df_rows: int, first_df_columns: int, changed_df_rows: int, changed_df_columns: int) -> None:
    """Prints the count of removed rows and columns.

    Args:
        first_df_rows (int): number of rows in dataframe before changes
        first_df_columns (int): number of columns in dataframe before changes
        changed_df_rows (int): number of rows in dataframe after changes
        changed_df_columns (int): number of columns in dataframe after changes
    """
    delta_df_rows, delta_df_columns = get_delta_shape(first_df_rows, first_df_columns, changed_df_rows, changed_df_columns)
    print(f'Removed {delta_df_rows} rows. Removed {delta_df_columns} columns.')


def measurement_conversion(df: pd.DataFrame, target_columns: typing.List[str], unit_column: str, orig_measurement_val: str, target_measurement_val: str, rationumber: float) -> pd.DataFrame:
    """Convert measurements in selected columns using a ratio when a specific unit matches. For example: Convert CM to MM by multiplying it by 10.

    Args:
        df (pd.DataFrame): dataframe object
        target_columns (typing.List[str]): list of strings, the target columns to be converted with a ratio
        unit_column (str): the column describing the unit
        orig_measurement_val (str): the string, value of the original unit measure
        target_measurement_val (str): the string, value of the converted unit measure
        rationumber (float): the ratio number for the conversion

    Returns:
        pd.DataFrame: updated dataframe
    """
    try:
        #Boolean mask for rows needing conversion
        mask = df[unit_column] == orig_measurement_val
        # Multiplying the target columns
        df.loc[mask, target_columns] = df.loc[mask, target_columns] * rationumber
        # Replacing unit value
        df.loc[mask, unit_column] = target_measurement_val
        print(f'Measurement conversion done on {unit_column}, from {orig_measurement_val} to {target_measurement_val}')
    except Exception as e:
        print(f'Issue with measurement conversion: {e}')

    return df


def clean_up_columns(df: pd.DataFrame, columns_to_remove: typing.List[str]) -> pd.DataFrame:
    try:
        df = df.drop(columns=columns_to_remove)
        print(f'Columns {columns_to_remove} removed from dataframe.')
    except Exception as e:
        print(f'Couldnt remove columns: {columns_to_remove} from dataframe.')
    return df


def convert_numeric_dtype(df: pd.DataFrame, num_cols=None):
    if num_cols is None:
        num_cols = []
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def drop_rows_by_value(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    return df[df[col] != value]


def drop_rows_by_values(df: pd.DataFrame, col: str, values: list[str] = []) -> pd.DataFrame:
    for value in values:
        df = drop_rows_by_value(df, col, value)
    return 


def clip_percentile(df: pd.DataFrame, columns: list, percentile: float):
    for column in columns:
        df[column] = df[column].clip(upper=df[column].quantile(0.99))
    return df


def clip_small_amounts(df, threshold, destination_column):
    # Get counts of each class
    counts = df[destination_column].value_counts()
    
    # Find classes that have counts below threshold
    small_classes = counts[counts < threshold].index.tolist()
    
    # Filter out rows with these small classes
    df_clipped = df[~df[destination_column].isin(small_classes)].copy()
    
    return df_clipped


def remove_impossible_values(df, min_dimension=MIN_DIMENSION, max_density=MAX_DENSITY, max_weight=MAX_WEIGHT):
    """
    Remove records with impossible physical values:
    - Any dimension below min_dimension
    - Density above max_density
    - Net weight above max_weight
    """
    df_clean = df[
        (df["Length_mm"] > min_dimension) &
        (df["Width_mm"] > min_dimension) &
        (df["Height_mm"] > min_dimension) &
        (df["Density_kg_m3"] < max_density) &
        (df["Net_Weight_kg"] < max_weight)
    ].copy()
    return df_clean


def winsorize_columns(df, columns, lower_pct=0, upper_pct=0.01):
    """
    Apply winsorization to cap extreme high values (upper_pct).
    Keeps most data but limits the influence of extreme outliers.
    """
    df_wins = df.copy()
    for col in columns:
        df_wins[col] = winsorize(df_wins[col], limits=(lower_pct, upper_pct))
    return df_wins


def log_transform_columns(df, columns):
    """
    Apply log(1+x) transformation to reduce skew in numeric features.
    """
    df_log = df.copy()
    for col in columns:
        df_log[col + "_log"] = np.log1p(df_log[col])
    return df_log


def clip_percentile(df, columns, percentile=0.99):
    """
    Clip numeric columns at the specified upper percentile.
    """
    df_clip = df.copy()
    for col in columns:
        upper_limit = df_clip[col].quantile(percentile)
        df_clip[col] = df_clip[col].clip(lower=None, upper=upper_limit)
    return df_clip


def full_cleaning_pipeline(df, numeric_cols=NUMERIC_COLUMNS_TO_CHANGE_NAMES):
    # 1. Remove impossible values
    df_clean = remove_impossible_values(df)
    
    # 2. Winsorize numeric features
    df_wins = winsorize_columns(df_clean, numeric_cols)
    
    # 3. Log-transform skewed features
    #skewed_cols = ["Net_Weight_kg", "Volume_m3", "Density_kg_m3"]
    #df_log = log_transform_columns(df_wins, skewed_cols)
    
    # 4. Clip 0.99 percentile
    #df_final = clip_percentile(df_log, numeric_cols)
    df_final = clip_percentile(df_wins, numeric_cols)
    
    return df_final
