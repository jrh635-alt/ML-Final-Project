#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:46:55 2026

@author: jack
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    return X_train, X_test, y_train, y_test

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

    # Data splitting
    X_train, X_test, y_train, y_test = split_data(df_features, features, target)

    # Linear regression
    y_pred, weights_df = linear_regression(X_train, X_test, y_train, y_test)

    print('***LINEAR REGRESSION***')
    print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

    weights_df['Abs_Weight'] = weights_df['Weight'].abs()
    weights_df = weights_df.sort_values(by='Abs_Weight', ascending=False).drop(
        columns=['Abs_Weight'])

    print(weights_df)

    #







