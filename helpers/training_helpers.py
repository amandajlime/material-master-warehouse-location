from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier


def scale_df(df, columns: list):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    return X_scaled


def randomforestclass(X_train, y_train, X_test, n_estimators: int, random_state: int, max_depth=None, max_features: str = None, min_samples_leaf: int = None, min_samples_split: int = None):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=random_state, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def oversample(X_train, y_train):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    return X_res, y_res


def searchcvgrid(X_train, y_train, X_test, y_test):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [200],  # already stable
        'max_depth': [None],    # leave fully grown
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced']  # keep balancing for minority classes
    }

    # Create the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1_weighted',  # prioritize overall performance
        cv=5,                   # 5-fold cross-validation
        n_jobs=-1,              # use all cores
        verbose=2
    )

    # Fit GridSearchCV on your training data
    grid_search.fit(X_train, y_train)

    # Print best hyperparameters
    print("Best hyperparameters found:")
    print(grid_search.best_params_)

    # Evaluate the best model on test set
    best_rf = grid_search.best_estimator_
    preds = best_rf.predict(X_test)

    print(classification_report(y_test, preds))


# Function to train and evaluate a BalancedRandomForest
def balanced_randomforest_model(X, y, test_size=0.3, random_state=42, n_estimators=200, max_depth=None):
    """
    Trains a BalancedRandomForestClassifier and prints evaluation metrics.
    
    Parameters:
    - X: features DataFrame
    - y: target Series
    - test_size: fraction of data to use as test
    - random_state: for reproducibility
    - n_estimators: number of trees
    - max_depth: maximum depth of trees
    """
    # 1. Split data with stratification to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Initialize BalancedRandomForestClassifier
    brf = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        replacement=False,   # default, can try True for sampling with replacement
        sampling_strategy='auto',  # balances each minority class to match the majority
    )
    
    # 3. Train the model
    brf.fit(X_train, y_train)
    
    # 4. Make predictions
    y_pred = brf.predict(X_test)
    
    # 5. Evaluation
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    
    return brf, X_train, X_test, y_train, y_test, y_pred
