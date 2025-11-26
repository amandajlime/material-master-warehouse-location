import pandas as pd
import typing


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
    return df