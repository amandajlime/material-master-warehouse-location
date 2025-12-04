import helpers.data_cleaning_helpers as data_clean
import helpers.data_plotting_helpers as data_plot
import helpers.data_transform_helpers as data_transform
import helpers.data_enrichment_helpers as data_enrich
import helpers.training_helpers as training
import helpers.accuracy_helpers as accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
from config import (
    CSV_SOURCE_STR,
    FEATHER_DEST_STR,
    DESTINATION_COLUMN,
    THRESHOLD_FOR_DROPPING,
    NO_NULL_COLUMNS,
    UNITS_TO_IGNORE,
    MEASUREMENT_CONVERSION,
    COLUMNS_TO_CLEANUP,
    NUMERIC_COLUMNS,
    EDGE_CASE_GROUPS_ALLDATA,
    EDGE_CASE_GROUPS_TRAINDATA,
    FEATURE_RENAMING,
    TRAININGCOLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_FEATURES,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    MIN_DIMENSION,
    MAX_DENSITY,
    MAX_WEIGHT,
    NUMERIC_COLUMNS_TO_CHANGE_NAMES)

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

# Convert numeric columns to numeric dtype
df = data_clean.convert_numeric_dtype(df, NUMERIC_COLUMNS)

# Convert measurements
for unit_column, values in MEASUREMENT_CONVERSION.items():
    target_columns = values.get('target_columns')
    rationumber = values.get('rationumber')
    orig_measurement_val = values.get('orig_measurement_val')
    target_measurement_val = values.get('target_measurement_val')
    df = data_clean.measurement_conversion(df, target_columns, unit_column, orig_measurement_val, target_measurement_val, rationumber)

# Clean-up unnecessary columns
df = data_clean.clean_up_columns(df, COLUMNS_TO_CLEANUP)

# Enrich with volume and density
df = data_enrich.enrich_with_volume_and_density(df)

# Rename features to include measurement units
for key, value in FEATURE_RENAMING.items():
    df = data_enrich.rename_features(df, key, value)

# Clean the dataframe with a cleaning pipeline
df_cleaned = data_clean.full_cleaning_pipeline(df)
df_cleaned = data_clean.clip_small_amounts(df_cleaned, 5, DESTINATION_COLUMN)

# Printing the description of the cleaned dataframe and the count of remaining classes - commented out
#print(df_cleaned.describe())
#print(f'COUNT OF CLASSES: {df_cleaned[DESTINATION_COLUMN].value_counts()}')

trainingcolumns = TRAININGCOLUMNS.get("trainingcolumns")

# Scaling the features
X_scaled = training.scale_df(df_cleaned, trainingcolumns)
X_scaled_df = pd.DataFrame(X_scaled, columns=trainingcolumns)

# Peeking at dataframe details - commented out
#accuracy.print_df_details(X_scaled_df)

# Take a look at scaled data - comment out when not in use
#X_scaled_df[DESTINATION_COLUMN] = df_cleaned[DESTINATION_COLUMN].values
#data_plot.plot_kde(X_scaled_df, hue=DESTINATION_COLUMN)

# Defining y, which is taken from the destination column in the cleaned dataframe
y = df_cleaned[DESTINATION_COLUMN]

# Split the data and stratify y
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Oversample the training data
X_oversampled, y_oversampled = training.oversample(X_train, y_train)

# Search the best hyperparameters to optimize the model - commented out because the best hyperparameters were added to environment variables
#training.searchcvgrid(X_oversampled, y_oversampled, X_test, y_test)

# Train the model with the best hyperparameters prioritizing f1-score
model, preds = training.randomforestclass(X_oversampled, y_oversampled, X_test, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, max_depth=None, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF, min_samples_split=MIN_SAMPLES_SPLIT)

# Evaluate the model
accuracy.print_classification_report(y_test, preds)
