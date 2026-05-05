#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:46:55 2026

@author: jack
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

def read_data(data_file_path, headers_file_path, features_file_path):
    df_data = pd.read_csv(data_file_path)
    headers = pd.read_csv(headers_file_path)
    df_features = pd.read_csv(features_file_path)

    df = pd.merge(df_data, df_features, left_on='fips_str2020', right_on='fips', how='inner')

    header_dict = dict(zip(headers['name'], headers['varlab']))
    df.rename(columns=header_dict, inplace=True)

    return df

def clean_data(df, features, target):
    df_features = df[features + [target]].dropna()
    return df_features

def split_data(df_features, features, target):

    X = df_features[features]
    y = df_features[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

    # Scaling to ensure certain features don't take over
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def baseline_regression(X_train, X_test, y_train, y_test, features):
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    return y_pred_baseline

def linear_regression(X_train, X_test, y_train, y_test, features):
    ULRmodel = LinearRegression()
    ULRmodel.fit(X_train, y_train)

    y_pred = ULRmodel.predict(X_test)

    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': ULRmodel.coef_
    })

    print(f"Intercept: {ULRmodel.intercept_}")

    return y_pred, weights_df

def sgd_regression(X_train, X_test, y_train, y_test, features):
    sgd_model = SGDRegressor(random_state=67)
    sgd_model.fit(X_train, y_train)
    y_pred = sgd_model.predict(X_test)

    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': sgd_model.coef_
    })

    print(f"Intercept: {sgd_model.intercept_}")
    return y_pred, weights_df

def ridge_regression(X_train, X_test, y_train, y_test, features):
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': ridge_model.coef_
    })
    print(f"Intercept: {ridge_model.intercept_}")
    return y_pred, weights_df

def lasso_regression(X_train, X_test, y_train, y_test, features):
    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': lasso_model.coef_
    })
    print(f"Intercept: {lasso_model.intercept_}")
    return y_pred, weights_df

def binary_preprocessing(y_train, y_test):
    binary_y_train = (y_train > 0).astype(int)
    binary_y_test = (y_test > 0).astype(int)
    return binary_y_train, binary_y_test

def logistic_regression(X_train, X_test, y_train, y_test, features):
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    proba = log_model.predict_proba(X_test)[:, 1]
    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': log_model.coef_[0]
    })
    weights_df['Abs_Weight'] = weights_df['Weight'].abs()
    weights_df = weights_df.sort_values(by='Abs_Weight', ascending=False)
    weights_df = weights_df.drop(columns=['Abs_Weight'])
    return y_pred, weights_df, proba

def decision_tree_regression(X_train, X_test, y_train, y_test, features):
    tree_model = DecisionTreeRegressor(random_state=67)
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Importance': tree_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return y_pred, weights_df


def random_forest_regression(X_train, X_test, y_train, y_test, features):
    rf_model = RandomForestRegressor(random_state=67)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    weights_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return y_pred, weights_df

def elastic_net_regression(X_train, X_test, y_train, y_test, features):
    elastic_net_model = ElasticNet()
    elastic_net_model.fit(X_train, y_train)
    y_pred = elastic_net_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Elastic_Net': elastic_net_model.coef_
    })
    return y_pred, weights_df

def random_forest_classification(X_train, X_test, y_train, y_test, features):
    rf_model = RandomForestClassifier(random_state=67)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    proba = rf_model.predict_proba(X_test)[:, 1]
    weights_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    })
    return y_pred, weights_df, proba

def gradient_boosting_regression(X_train, X_test, y_train, y_test, features):
    gbm_model = GradientBoostingRegressor(random_state=67)
    gbm_model.fit(X_train, y_train)
    y_pred = gbm_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Importance': gbm_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return y_pred, weights_df

def gradient_boosting_classification(X_train, X_test, y_train, y_test, features):
    gbm_model = GradientBoostingClassifier()
    gbm_model.fit(X_train, y_train)
    y_pred = gbm_model.predict(X_test)
    proba = gbm_model.predict_proba(X_test)[:, 1]
    weights_df = pd.DataFrame({
        'Feature': features,
        'Importance': gbm_model.feature_importances_
    })
    return y_pred, weights_df, proba

def decision_tree_classification(X_train, X_test, y_train, y_test, features):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    proba = dt_model.predict_proba(X_test)[:, 1]
    weights_df = pd.DataFrame({
        'Feature': features,
        'Importance': dt_model.feature_importances_
    })
    return y_pred, weights_df, proba


def svm_classification(X_train, X_test, y_train, y_test, features):
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    proba = svm_model.predict_proba(X_test)[:, 1]
    y_pred = svm_model.predict(X_test)

    return y_pred, proba


if __name__ == '__main__':
    # File paths
    data_file_path = 'data/uncleaneddata.csv'
    headers_file_path = 'data/headers.csv'
    features_file_path = 'data/uncleanedfeatures.csv'

    # Reading and cleaning
    df_data = read_data(data_file_path, headers_file_path, features_file_path)
    print(f'Data read in: \n{df_data.head()}')
    features = [
        "p_hisp_2010", "p_child_hh_2010", "h_units_2010", "p_male_2010",
        "med_age_2010", "p_asian_2010", "p_black_2010", "p_white_2010",
        "p_65plus_2010", "p_ownerocc_2010", "pop_2010", "p_season_2010",
        "p_singpar_2010", "p_under18_2010", "p_vacant_2010",
        "metromicronon_2020",
        "farming_2015", "mining_2015", "manufacturing_2015", "government_2015",
        "recreation_2015", "nonspecialized_2015", "low_education_2015",
        "low_employment_2008_2012", "pop_loss", "retirement_dest_2015",
        "persistent_poverty_2013", "persistent_child_poverty_2013", "rucc_2023"
    ]
    target = 'Net Migration Rate 2010s, total'
    df_features = clean_data(df_data, features, target)
    print(f'df_features: \n {df_features.head()}')
    print(f'df_features.shape: {df_features.shape}')

    # Plotting target distribution to check for skewness
    plt.figure()
    sns.histplot(df_features[target])
    plt.title('Distribution of Target Variable')
    plt.show()

    '''
    # Histograms to visualize spread within each feature
    for feature in features:
       plt.figure()
       sns.histplot(df_features[feature])
       plt.title(feature)
       plt.show()
    '''

    # Time permitting - k fold CV?

    # Data splitting
    X_train, X_test, y_train, y_test = split_data(df_features, features, target)

    # Prep for binary classification methods
    binary_y_train, binary_y_test = binary_preprocessing(y_train, y_test)

    regression_models = [linear_regression, sgd_regression, ridge_regression,
                         lasso_regression, random_forest_regression,
                         elastic_net_regression, gradient_boosting_regression]
    classification_models = [logistic_regression, decision_tree_classification,
                             random_forest_classification, gradient_boosting_classification,
                             svm_classification]
    tree_models = [decision_tree_regression, decision_tree_classification,
                   random_forest_regression, random_forest_classification,
                   gradient_boosting_regression, gradient_boosting_classification]

    results_df_regression = pd.DataFrame(columns=['Actual', 'Predicted', 'importance/weight', 'R^2', 'MSE'])

    for regression_model in regression_models:
        y_pred_reg, weights_df_reg = regression_model(X_train, X_test,
                                                       y_train, y_test,
                                                       features)
        weights_df_reg['Abs_Weight'] = weights_df_reg['Weight'].abs()
        weights_df_reg= weights_df_reg.sort_values(by='Abs_Weight',
                                                    ascending=False).drop(
            columns=['Abs_Weight'])
        results_df_regression.loc[regression_model] = [y_test, y_pred_reg, weights_df_reg,
                                                       r2_score(y_test, y_pred_reg),
                                                       mean_squared_error(y_test, y_pred_reg)]


    results_df_classification = pd.DataFrame(columns=['Actual','Predicted', 'importance/weight',
                                                      'F', 'Accuracy', 'Precision',
                                                      'Recall', 'ROC'])
    for classification_model in classification_models:
        y_pred_clf, weights_clf, proba = classification_model(X_train, X_test,
                                                       binary_y_train, binary_y_test,
                                                       features)
        results_df_classification[classification_model] = [binary_y_test, y_pred_clf, weights_clf,
                                                           f1_score(binary_y_test, y_pred_clf),
                                                           accuracy_score(binary_y_test, y_pred_clf),
                                                           precision_score(binary_y_test, y_pred_clf),
                                                           recall_score(binary_y_test, y_pred_clf),
                                                           roc_auc_score(y_test, proba)]

    results_df_tree = pd.DataFrame(columns=['Actual','Predicted', 'importance/weight',])


    results_dfs = [results_df_regression, results_df_classification, results_df_tree]

    for results_df in results_dfs:
        results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
        results_df['Abs_Residual'] = results_df['Residual'].abs()
        print('***ERROR ANALYSIS: TOP 10 Largest Residuals***')
        print(results_df.sort_values(by='Abs_Residual', ascending=False).head(10))

    # Decision tree regression
    y_pred_dtree, weights_dtree = decision_tree_regression(X_train, X_test, y_train, y_test, features)
    print('***DECISION TREE REGRESSION***')
    print(f"R^2 Score: {r2_score(y_test, y_pred_dtree):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_dtree):.4f}")

    # Decision tree classification
    y_pred_dtree_c, weights_dtree_c = decision_tree_classification(X_train, X_test, binary_y_train, binary_y_test, features)
    print('***DECISION TREE CLASSIFICATION***')
    print(f"Accuracy: {accuracy_score(binary_y_test, y_pred_rf_clf):.4f}")
    print(f"F1 Score: {f1_score(binary_y_test, y_pred_rf_clf):.4f}")
    print(weights_dtree_c)

    # Hyperparameter tuning on a few of the better models?


    # Plot residuals at some point?




