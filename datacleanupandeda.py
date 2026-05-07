#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:46:55 2026

@author: jack
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
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
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time



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
        'Weight': elastic_net_model.coef_
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
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)
    proba = svm_model.predict_proba(X_test)[:, 1]
    y_pred = svm_model.predict(X_test)

    return y_pred, None, proba


def plot_regression_metrics(results_df_regression):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    r2_sorted = results_df_regression['R^2'].sort_values(ascending=False)
    mse_sorted = results_df_regression['MSE'].sort_values(ascending=True)

    axes[0].barh(r2_sorted.index, r2_sorted.values)
    axes[0].set_title('R² by Model')
    axes[0].set_xlabel('R²')

    axes[1].barh(mse_sorted.index, mse_sorted.values)
    axes[1].set_title('MSE by Model')
    axes[1].set_xlabel('MSE')

    plt.tight_layout()
    plt.savefig('plots/regression_metrics.png')
    plt.show()


def plot_classification_metrics(results_df_classification):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    roc_sorted = results_df_classification['ROC-AUC'].sort_values(ascending=False)
    f1_sorted = results_df_classification['F'].sort_values(ascending=False)

    axes[0].barh(roc_sorted.index, roc_sorted.values)
    axes[0].set_title('ROC-AUC by Model')
    axes[0].set_xlabel('ROC-AUC')

    axes[1].barh(f1_sorted.index, f1_sorted.values)
    axes[1].set_title('F1 by Model')
    axes[1].set_xlabel('F1')

    for model_name in results_df_classification.index:
        actual = results_df_classification.loc[model_name, 'Actual']
        proba = results_df_classification.loc[model_name, 'Proba']
        rocauc = results_df_classification.loc[model_name, 'ROC-AUC']
        fpr, tpr, _ = roc_curve(actual, proba)
        axes[2].plot(fpr, tpr, label=f'{model_name} (AUC={rocauc:.3f})')

    axes[2].plot([0, 1], [0, 1], 'k--', label='Random classifier')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC Curves')
    axes[2].legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig('plots/classification_metrics.png')
    plt.show()


def plot_feature_importance(results_df, model_name):
    weights_df = results_df.loc[model_name, 'importance/weight']
    if weights_df is None:
        return

    weight_col = 'Weight' if 'Weight' in weights_df.columns else 'Importance'

    plt.figure(figsize=(10, 8))
    colors = ['fuchsia' if w < 0 else 'green' for w in weights_df[weight_col]]
    plt.barh(weights_df['Feature'], weights_df[weight_col], color=colors)
    plt.title(f'Feature Importance/Weight: {model_name}')
    plt.xlabel(weight_col)
    plt.tight_layout()
    plt.savefig(f'plots/importance_{model_name}.png')
    plt.show()


def plot_residuals(results_df_regression):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, model_name in enumerate(results_df_regression.index):
        actual = results_df_regression.loc[model_name, 'Actual']
        predicted = results_df_regression.loc[model_name, 'Predicted']
        residuals = actual - predicted

        axes[i].scatter(predicted, residuals, alpha=0.3, s=10)
        axes[i].axhline(y=0, color='fuchsia', linestyle='--')
        axes[i].set_title(model_name)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Residual')

    plt.tight_layout()
    plt.savefig('plots/residuals.png')
    plt.show()


def plot_actual_vs_predicted(results_df_regression):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, model_name in enumerate(results_df_regression.index):
        actual = results_df_regression.loc[model_name, 'Actual']
        predicted = results_df_regression.loc[model_name, 'Predicted']

        axes[i].scatter(actual, predicted, alpha=0.3, s=10)
        axes[i].plot([actual.min(), actual.max()],
                     [actual.min(), actual.max()], 'r--')
        axes[i].set_title(model_name)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')

    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png')
    plt.show()

def hyperparameter_tuning(results_df, model_name, model_function,
                          tuning_times, params, grid_or_random, reg_or_class,
                          X_train, X_test, y_train, y_test):
    t_start = time.perf_counter()
    if grid_or_random == "grid" and reg_or_class == "reg":
        search = GridSearchCV(model_function(random_state=67), params,
                               cv=5, n_jobs=-1, scoring='r2')
    elif grid_or_random == "random" and reg_or_class == "reg":
        search = RandomizedSearchCV(model_function(random_state=67),
                                    params, n_iter=20, cv=5, n_jobs=-1,
                                    scoring='r2', random_state=67)
    elif grid_or_random == "grid" and reg_or_class == "class":
        search = GridSearchCV(model_function(random_state=67),
                              params, cv=5, n_jobs=-1, scoring='roc_auc')

    elif grid_or_random == "random" and reg_or_class == "class":
        search = RandomizedSearchCV(model_function(random_state=67),params,
                                    n_iter=20, cv=5, n_jobs=-1, scoring='roc_auc',
                                    random_state=67)

    search.fit(X_train, y_train)
    tuning_times[f'{model_name}_{grid_or_random}'] = time.perf_counter() - t_start

    best_model = search.best_estimator_

    best_model.fit(X_train, y_train)
    y_pred_retrain = best_model.predict(X_test)
    imp_df = pd.DataFrame({'Feature': features,
                           'Importance': best_model.feature_importances_})
    imp_df['Abs_Weight'] = imp_df['Importance'].abs()
    imp_df = imp_df.sort_values('Abs_Weight', ascending=False).drop(
        columns='Abs_Weight')
    if reg_or_class == "class":
        proba = best_model.predict_proba(X_test)[:, 1]
        results_df.loc[f'{model_name} {grid_or_random} search'] = [
            y_test, y_pred_clf, imp_df,
            f1_score(y_test, y_pred_clf),
            accuracy_score(y_test, y_pred_clf),
            precision_score(y_test, y_pred_clf),
            recall_score(y_test, y_pred_clf),
            roc_auc_score(y_test, proba),
            proba
        ]


    if reg_or_class == "reg":
        results_df.loc[f'{model_name} {grid_or_random} search'] = [y_test, y_pred_retrain, imp_df,
                                                                          r2_score(y_test, y_pred_retrain),
                                                                          mean_squared_error(y_test, y_pred_retrain)]
    return tuning_times, results_df


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

    # =======================
    # DATA SUMMARY
    # =======================

    print("\n***DATA SUMMARY***")

    # Number of samples and features
    print("\nNumber of samples:", df_features.shape[0])
    print("Number of features:", df_features.shape[1] - 1)

    # -----------------------
    # Target summary
    # -----------------------
    print("\nTarget Summary:")
    summary = df_features[target].describe()
    summary_df = summary.to_frame(name='Value')
    summary_df.index.name = 'Statistic'
    print(summary_df)

    # -----------------------
    # Feature summary (sample)
    # -----------------------
    print("\nFeature Summary (sample of features):")
    feature_summary = df_features[features].describe().T[['mean', 'std', 'min', 'max']]
    print(feature_summary.head())

    # -----------------------
    # Binary distribution
    # -----------------------
    binary = (df_features[target] > 0).astype(int)
    print("\nBinary Distribution:")
    print(binary.value_counts(normalize=True))

    # -----------------------
    # Plot target distribution
    # -----------------------
    plt.figure()
    sns.histplot(df_features[target])
    plt.title('Distribution of Net Migration Rate')
    plt.xlabel('Net Migration Rate')
    plt.ylabel('Count')
    plt.show()

    '''
    # Plotting target distribution to check for skewness
    plt.figure()
    sns.histplot(df_features[target])
    plt.title('Distribution of Target Variable')
    plt.show()
    '''

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
                         gradient_boosting_regression, decision_tree_regression]
    classification_models = [logistic_regression, decision_tree_classification,
                             random_forest_classification, gradient_boosting_classification]
    tree_models = [decision_tree_regression, decision_tree_classification,
                   random_forest_regression, random_forest_classification,
                   gradient_boosting_regression, gradient_boosting_classification,
                   svm_classification]

    results_df_regression = pd.DataFrame(columns=['Actual', 'Predicted', 'importance/weight', 'R^2', 'MSE'])

    for regression_model in regression_models:
        y_pred_reg, weights_df_reg = regression_model(X_train, X_test,
                                                      y_train, y_test,
                                                      features)

        weight_col = 'Weight' if 'Weight' in weights_df_reg.columns else 'Importance'
        weights_df_reg['Abs_Weight'] = weights_df_reg[weight_col].abs()
        weights_df_reg = weights_df_reg.sort_values(by='Abs_Weight', ascending=False).drop(columns=['Abs_Weight'])

        results_df_regression.loc[regression_model.__name__] = [y_test, y_pred_reg,
                                                                weights_df_reg,
                                                                r2_score(y_test, y_pred_reg),
                                                                mean_squared_error(y_test,y_pred_reg)]


    results_df_classification = pd.DataFrame(columns=['Actual','Predicted', 'importance/weight',
                                                      'F', 'Accuracy', 'Precision',
                                                      'Recall', 'ROC-AUC', 'Proba'])
    for classification_model in classification_models:
        y_pred_clf, weights_clf, proba = classification_model(X_train, X_test,
                                                       binary_y_train, binary_y_test,
                                                       features)
        results_df_classification.loc[classification_model.__name__] = [binary_y_test, y_pred_clf, weights_clf,
                                                           f1_score(binary_y_test, y_pred_clf),
                                                           accuracy_score(binary_y_test, y_pred_clf),
                                                           precision_score(binary_y_test, y_pred_clf),
                                                           recall_score(binary_y_test, y_pred_clf),
                                                           roc_auc_score(binary_y_test, proba),
                                                                        proba]


    y_pred_elastic, weights_elastic = elastic_net_regression(X_train, X_test, y_train, y_test, features)
    results_df_regression.loc[elastic_net_regression.__name__] = [y_test, y_pred_elastic, weights_elastic,
                                                         r2_score(y_test, y_pred_elastic),
                                                         mean_squared_error(y_test, y_pred_elastic)]

    y_pred_baseline = baseline_regression(X_train, X_test, y_train, y_test, features)
    results_df_regression.loc[baseline_regression.__name__] = [y_test, y_pred_baseline, None,
                                                        r2_score(y_test, y_pred_baseline),
                                                        mean_squared_error(y_test, y_pred_baseline)]


    results_dfs = [results_df_regression, results_df_classification]
    pd.set_option('display.max_columns', None)

    print('***REGRESSION STATS***')
    print(results_df_regression[['R^2', 'MSE']].sort_values(by='R^2', ascending=False))


    print('***CLASSIFICATION STATS***')
    print(results_df_classification[['F', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']].sort_values(by='ROC-AUC', ascending=False))


    print('***REGRESSION DF***')
    print(results_df_regression.head(10))

    print('***CLASSIFICATION DF***')
    print(results_df_classification.head(10))

    # Plots - for stats
    os.makedirs('plots', exist_ok=True)
    plot_regression_metrics(results_df_regression)
    plot_classification_metrics(results_df_classification)
    plot_actual_vs_predicted(results_df_regression)

    # Plots - features
    # Importances are always positive so we have them in green,
    # The line really only tells us about weights
    # skip baseline and svm
    for model_name in results_df_regression.index:
        plot_feature_importance(results_df_regression, model_name)

    for model_name in results_df_classification.index:
        plot_feature_importance(results_df_classification, model_name)


    # plot residuals - predicted on x, actual on y
    plot_residuals(results_df_regression)

    '''for results_df in results_dfs:
        results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
        results_df['Abs_Residual'] = results_df['Residual'].abs()
        print('***ERROR ANALYSIS: TOP 10 Largest Residuals***')
        print(results_df.sort_values(by='Abs_Residual', ascending=False).head(10))'''

    # Hyperparameter tuning on a few of the better models?
    tuning_times = {}
    # Gradient boosting regression
    # Grid search
    # log start time
    gbr_params_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }

    tuning_times, results_df_regression = hyperparameter_tuning(results_df_regression,
                                                                'gbr', GradientBoostingRegressor,
                                                                tuning_times, gbr_params_grid, 'grid',
                                                                'reg', X_train, X_test, y_train, y_test)
    print('Done with grid search GBR!')
    # Random search
    gbr_params_random = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
    }
    tuning_times, results_df_regression = hyperparameter_tuning(results_df_regression, 'gbr',
                                                                GradientBoostingRegressor,
                                                                tuning_times, gbr_params_random,
                                                                'random', 'reg',
                                                                X_train, X_test, y_train, y_test)

    print('Done with random search GBR!')
    # Random forest regression
    rfr_grid_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5],
    }
    tuning_times, results_df_regression = hyperparameter_tuning(results_df_regression, 'rfr',
                                                                RandomForestRegressor,
                                                                tuning_times, rfr_grid_params,
                                                                'grid', 'reg',
                                                                X_train, X_test, y_train, y_test)
    print('Done with grid search RFR!')
    rfr_random_params = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    tuning_times, results_df_regression = hyperparameter_tuning(
        results_df_regression, 'rfr',
        RandomForestRegressor,
        tuning_times, rfr_random_params,
        'random', 'reg', X_train, X_test, y_train, y_test)

    print('Done with random search RFR!')
    # Gradient boosting classification
    gbc_grid_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
    }

    tuning_times, results_df_classification = hyperparameter_tuning(results_df_classification,
                                                                    'gbc', GradientBoostingClassifier,
                                                                    tuning_times, gbc_grid_params, 'grid', 'class',
                                                                    X_train, X_test, binary_y_train, binary_y_test)
    print('Done with grid search GBC!')
    gbc_random_params = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
    }

    tuning_times, results_df_classification = hyperparameter_tuning(results_df_classification, 'gbc',
                                                                    GradientBoostingClassifier,
                                                                    tuning_times, gbc_random_params, 'random', 'class',
                                                                    X_train, X_test, binary_y_train, binary_y_test)

    print('Done with random search GBC!')

    # Look at stats and plots again
    print('\n***REGRESSION STATS (including tuned)***')
    print(results_df_regression[['R^2', 'MSE']].sort_values(by='R^2',
                                                            ascending=False))

    print('\n***CLASSIFICATION STATS (including tuned)***')
    print(results_df_classification[
              ['F', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']
          ].sort_values(by='ROC-AUC', ascending=False))

    print(tuning_times)







