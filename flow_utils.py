"""

    This script collects utility functions to make the flow file smaller and more readable.

"""


import pandas as pd
import numpy as np

def data(df):
    cat = df.select_dtypes(include='object')
    num = df.select_dtypes(exclude='object')
    return cat, num


def drop_unique(cat, df):
    """
        drop unique features in categorical columns
    """
    unique_value = cat.describe().columns[cat.describe().loc['unique'] == 1]
    if len(unique_value) > 0:
        df = df.drop(list(unique_value), axis = 1)
    else:
        print("No unique value found")        
        # get new categorical columns
    cat = df.select_dtypes(include='object')
    return cat, df

def cat_convert_num(categorical_columns, df):
    from sklearn.preprocessing import LabelEncoder
    encoder_list = []
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    # Convert categorical columns to numeric labels
    for col in categorical_columns:
        label_encoder = label_encoder.fit(df[col])
        encoding = {}
        for i in list(label_encoder.classes_):
            encoding[i]=label_encoder.transform([i])[0]
        df[col] = label_encoder.transform(df[col])
        encoder_list.append(encoding)
    
    return encoder_list, df


def get_X_y(df):
    # drop ID as it is unnecessary
    X = df.drop('ID', axis = 1)
    # get X and y
    y = df['Loan Status']
    X = X.drop('Loan Status',axis=1)
    return X, y

def imbalance(percent, X, y):
    """
        make imbalanced target value balanced
    """
    from imblearn.over_sampling import SMOTE
    if (percent[1] >= 70) or (percent[1] <= 30):
        # use smote to balance the data
        smote=SMOTE()
        smote.fit(X,y)
        X,y=smote.fit_resample(X,y)
    return X, y

def select_model(models, X_test, y_test):
    from sklearn.metrics import classification_report
    precision = 0
    i = 0
    for model in models:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=3, output_dict = True)
        # as we care about who will default the loan, so we care about the precision on target value 1
        precision_1 = report['1']['precision']
        if precision_1 > precision:
            best_model = model
            model_report = report
            precision = precision_1
            # check bug
            print("this is the better model:", i)
            print("better model:",model)
        i += 1
    
    return best_model, model_report

def new_features(best_model, X, X_train, X_test):
    # Get feature importances from the model
    feature_importance = best_model.feature_importances_
    # Create a DataFrame to store feature names and their importances
    df_importance = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
    # get the index number of unimportant features
    filtered_rows = df_importance[df_importance['importance'] < 0.0005].index
    useless_feature = list(filtered_rows)

    # contains new X_train and X_test with modified features
    X_newtrain = np.delete(X_train, useless_feature, axis=1)
    X_newtest = np.delete(X_test, useless_feature, axis=1)
    new_X = X.drop(X.iloc[:, useless_feature], axis=1)

    return X_newtrain, X_newtest, new_X

def new_feature_model(X_newtrain,y_train,X_newtest,y_test, new_X):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    new_rf = RandomForestClassifier(random_state=42)
    new_rf.fit(X_newtrain, y_train)
    y_pred = new_rf.predict(X_newtest)

    # give feature names to model, store it for future use
    new_rf.feature_names = new_X.columns.tolist()

    report_rf = classification_report(y_test, y_pred, digits=3, output_dict = True)
    return new_rf, report_rf

def improved_model(X_newtrain, y_train, X_newtest, y_test, X):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import classification_report

    param_dist = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        }
    # RandomForestClassifier is the best model as we tested before
    rf = RandomForestClassifier(random_state=42)
    # Define a custom scoring function using precision for class 1
    precision_scorer = make_scorer(precision_score, pos_label=1)
    # Random search of parameters using 2 fold cross validation
    rf_random = RandomizedSearchCV(estimator=rf, 
                                param_distributions=param_dist, 
                                n_iter=50, cv=2, verbose=2, 
                                random_state=42, n_jobs=-1,
                                scoring=precision_scorer)
    # Fit the random search model
    rf_random.fit(X_newtrain, y_train)
    best_model = rf_random.best_estimator_
    best_model.feature_names = X.columns.tolist()
    y_pred = best_model.predict(X_newtest)
    best_report = classification_report(y_test, y_pred, digits=3, output_dict = True)
    return best_model, best_report
        