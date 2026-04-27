#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:46:55 2026

@author: jack
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC


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

def linear_regression(X_train, X_test, y_train, y_test):
    ULRmodel = LinearRegression()
    ULRmodel.fit(X_train, y_train)

    y_pred = ULRmodel.predict(X_test)

    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': ULRmodel.coef_
    })

    print(f"Intercept: {ULRmodel.intercept_}")

    return y_pred, weights_df

def sgd_regression(X_train, X_test, y_train, y_test):
    sgd_model = SGDRegressor()
    sgd_model.fit(X_train, y_train)
    y_pred = sgd_model.predict(X_test)

    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': sgd_model.coef_
    })

    print(f"Intercept: {sgd_model.intercept_}")
    return y_pred, weights_df

def ridge_regression(X_train, X_test, y_train, y_test):
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': ridge_model.coef_
    })
    print(f"Intercept: {ridge_model.intercept_}")
    return y_pred, weights_df

def lasso_regression(X_train, X_test, y_train, y_test):
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

def logistic_regression(X_train, X_test, y_train, y_test):
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    weights_df = pd.DataFrame({
        'Feature': features,
        'Weight': log_model.coef_[0]
    })
    print(f"Intercept: {log_model.intercept_[0]}")
    return y_pred, weights_df


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

    # Linear regression
    y_pred_lin, weights_df_lin = linear_regression(X_train, X_test, y_train, y_test)

    print('***LINEAR REGRESSION***')
    print(f"R^2 Score: {r2_score(y_test, y_pred_lin):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_lin):.4f}")

    weights_df_lin['Abs_Weight'] = weights_df_lin['Weight'].abs()
    weights_df_lin = weights_df_lin.sort_values(by='Abs_Weight', ascending=False).drop(
        columns=['Abs_Weight'])

    print(weights_df_lin)

    # SGD regression
    y_pred_sgd, weights_df_sgd = sgd_regression(X_train, X_test, y_train, y_test)

    print('***SGD REGRESSION***')
    print(f"R^2 Score: {r2_score(y_test, y_pred_sgd):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_sgd):.4f}")

    weights_df_sgd['Abs_Weight'] = weights_df_sgd['Weight'].abs()
    weights_df_sgd = weights_df_sgd.sort_values(by='Abs_Weight', ascending=False).drop(
        columns=['Abs_Weight'])

    print(weights_df_sgd)

    # Ridge Regression
    y_pred_ridge, weights_df_ridge = ridge_regression(X_train, X_test, y_train,
                                                y_test)

    print('***RIDGE REGRESSION***')
    print(f"R^2 Score: {r2_score(y_test, y_pred_ridge):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}")

    weights_df_ridge['Abs_Weight'] = weights_df_ridge['Weight'].abs()
    weights_df_ridge = weights_df_ridge.sort_values(by='Abs_Weight',
                                                ascending=False).drop(
        columns=['Abs_Weight'])

    print(weights_df_ridge)

    # Lasso Regression
    y_pred_lasso, weights_df_lasso = lasso_regression(X_train, X_test, y_train, y_test)
    print('***LASSO REGRESSION***')
    print(f"R^2 Score: {r2_score(y_test, y_pred_lasso):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_lasso):.4f}")

    weights_df_lasso['Abs_Weight'] = weights_df_lasso['Weight'].abs()
    weights_df_lasso = weights_df_lasso.sort_values(by='Abs_Weight', ascending=False).drop(
        columns=['Abs_Weight'])

    print(weights_df_lasso)

    # Prep for binary classification methods
    binary_y_train, binary_y_test = binary_preprocessing(y_train, y_test)
    print(f'binary_y_train: {binary_y_train.shape}')
    print(f'binary_y_test: {binary_y_test.shape}')

    # Logistic regression
    y_pred_logistic, weights_df_logistic = logistic_regression(X_train, X_test, binary_y_train, binary_y_test)
    print('***LOGISTIC REGRESSION***')
    print(f"R^2 Score: {r2_score(binary_y_test, y_pred_logistic):.4f}")
    print(f"MSE: {mean_squared_error(binary_y_test, y_pred_logistic):.4f}")
    print(weights_df_logistic)

    # Hyperparameter tuning on a few of the better models?


    # Plot residuals at some point?




