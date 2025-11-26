import helpers.data_cleaning_helpers as data_clean
import helpers.data_plotting_helpers as data_plot
import helpers.data_transform_helpers as data_transform
from dotenv import load_dotenv
import os
import json

load_dotenv()

# STARTING WITH GLOBAL VARIABLES FROM ENV
CSV_SOURCE_STR = os.getenv('CSV_SOURCE_STR')
FEATHER_DEST_STR = os.getenv('FEATHER_DEST_STR')
DESTINATION_COLUMN = os.getenv('DESTINATION_COLUMN')
THRESHOLD_FOR_DROPPING = int(os.getenv('THRESHOLD_FOR_DROPPING'))
NO_NULL_COLUMNS = json.loads(os.getenv('NO_NULL_COLUMNS'))
UNITS_TO_IGNORE = json.loads(os.getenv('UNITS_TO_IGNORE'))
MEASUREMENT_CONVERSION = json.loads(os.getenv('MEASUREMENT_CONVERSION'))
COLUMNS_TO_CLEANUP = json.loads(os.getenv('COLUMNS_TO_CLEANUP'))
NUMERIC_COLUMNS = json.loads(os.getenv('NUMERIC_COLUMNS'))
EDGE_CASE_GROUPS_ALLDATA = json.loads(os.getenv('EDGE_CASE_GROUPS_ALLDATA'))
EDGE_CASE_GROUPS_TRAINDATA = json.loads(os.getenv('EDGE_CASE_GROUPS_TRAINDATA'))

# Loading data into a pandas dataframe
df = data_transform.csv_to_feather(CSV_SOURCE_STR, FEATHER_DEST_STR, 0, ';')

# Remove rows where destination column has an empty value
df = data_clean.drop_na_rows(df, [DESTINATION_COLUMN])

# Remove rows where destination column's category is below the threshold
df = data_clean.drop_below_threshold_rows(df, DESTINATION_COLUMN, THRESHOLD_FOR_DROPPING)

# Remove rows from NO_NULL_COLUMNS that have empty values in them
df = data_clean.drop_na_rows(df, NO_NULL_COLUMNS)

# Remove rows that have UNITS_TO_IGNORE
# meaning, certain rows with certain units that ought to be removed
for col, units in UNITS_TO_IGNORE.items():
    for unit in units:
        df = data_clean.drop_str_match_rows(df, col, unit)

print(df.describe())

# Convert numeric columns to numeric dtype
df = data_clean.convert_numeric_dtype(df, NUMERIC_COLUMNS)

print(df.describe())

# Convert measurements
for unit_column, values in MEASUREMENT_CONVERSION.items():
    target_columns = values.get('target_columns')
    rationumber = values.get('rationumber')
    orig_measurement_val = values.get('orig_measurement_val')
    target_measurement_val = values.get('target_measurement_val')
    df = data_clean.measurement_conversion(df, target_columns, unit_column, orig_measurement_val, target_measurement_val, rationumber)

# Clean-up unnecessary columns
df = data_clean.clean_up_columns(df, COLUMNS_TO_CLEANUP)

print(df.describe())

# Drop some values
# df = data_clean.drop_rows_by_values(df, DESTINATION_COLUMN, EDGE_CASE_GROUPS_TRAINDATA)
# print(df.describe())

# Take a look at data
data_plot.plot_kde(df, hue=DESTINATION_COLUMN)

# Take another look at data
data_plot.scatter_matrix(df)
