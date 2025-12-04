from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


def print_classification_report(y_test, preds):
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))


def print_df_details(df: pd.DataFrame):
    print(df.head())
    print(df.describe())
