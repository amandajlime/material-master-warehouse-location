from dotenv import load_dotenv
import os
import json

load_dotenv()

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
FEATURE_RENAMING = json.loads(os.getenv('FEATURE_RENAMING'))
TRAININGCOLUMNS = json.loads(os.getenv("TRAININGCOLUMNS"))
TEST_SIZE = float(os.getenv('TEST_SIZE'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE'))
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS'))
MAX_FEATURES = os.getenv('MAX_FEATURES')
MIN_SAMPLES_LEAF = int(os.getenv('MIN_SAMPLES_LEAF'))
MIN_SAMPLES_SPLIT = int(os.getenv('MIN_SAMPLES_SPLIT'))
MIN_DIMENSION = int(os.getenv('MIN_DIMENSION'))
MAX_DENSITY = int(os.getenv('MAX_DENSITY'))
MAX_WEIGHT = int(os.getenv('MAX_WEIGHT'))
NUMERIC_COLUMNS_TO_CHANGE_NAMES = json.loads(os.getenv('NUMERIC_COLUMNS_TO_CHANGE_NAMES'))
