import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


def smart_imputation(X):
    """
    Smart imputation strategy based on feature characteristics
    """
    X_imputed = X.copy()

    # Identify feature types
    spectral_bands = [col for col in X.columns if col.startswith('B') and len(col) == 6]
    indices = [col for col in X.columns if col in ['NDVI', 'NDBI', 'NDWI', 'SAVI', 'UI', 'EBBI', 'IBI']]
    ratios = [col for col in X.columns if 'ratio' in col or 'div' in col]
    other_features = [col for col in X.columns if col not in spectral_bands + indices + ratios]

    # Different imputation strategies
    imputers = {
        'spectral': SimpleImputer(strategy='median'),
        'indices': SimpleImputer(strategy='mean'),
        'ratios': SimpleImputer(strategy='median'),
        'other': SimpleImputer(strategy='most_frequent')
    }

    # Apply imputation
    for feature_type, features in [('spectral', spectral_bands),
                                  ('indices', indices),
                                  ('ratios', ratios),
                                  ('other', other_features)]:
        if features:
            imputer = imputers[feature_type]
            X_imputed[features] = imputer.fit_transform(X[features])

    return X_imputed

def create_advanced_features(X):
    """
    Create advanced features for urban expansion prediction
    """
    X_advanced = X.copy()

    # 1. Basic Spectral Indices (if not already present)
    if all(col in X.columns for col in ['B4_red', 'B5_nir', 'B6_swir1', 'B3_green']):
        # Urban-specific indices
        X_advanced['UI'] = (X['B6_swir1'] - X['B5_nir']) / (X['B6_swir1'] + X['B5_nir'])  # Urban Index
        X_advanced['EBBI'] = (X['B6_swir1'] - X['B5_nir']) / (X['B6_swir1'] + X['B5_nir'] + X['B4_red'])  # Enhanced Built-up Index
        X_advanced['IBI'] = (2 * X['B6_swir1'] / (X['B6_swir1'] + X['B5_nir']) -
                            (X['B5_nir'] / (X['B5_nir'] + X['B4_red']) +
                             X['B4_red'] / (X['B4_red'] + X['B3_green']))) / \
                           (2 * X['B6_swir1'] / (X['B6_swir1'] + X['B5_nir']) +
                            (X['B5_nir'] / (X['B5_nir'] + X['B4_red']) +
                             X['B4_red'] / (X['B4_red'] + X['B3_green'])))  # Index-based Built-up Index

    # 2. Band Ratios and Combinations
    band_combinations = {
        'ratio_swir2_nir': 'B7_swir2 / B5_nir',
        'ratio_red_blue': 'B4_red / B2_blue',
        'ratio_green_blue': 'B3_green / B2_blue',
        'ratio_swir1_blue': 'B6_swir1 / B2_blue',
        'sum_swir_bands': 'B6_swir1 + B7_swir2',
        'diff_nir_red': 'B5_nir - B4_red',
        'brightness_index': '(B4_red + B5_nir + B6_swir1) / 3'
    }

    for new_feat, expression in band_combinations.items():
        try:
            X_advanced[new_feat] = X.eval(expression)
        except:
            pass

    # 3. Statistical Features (if we have spatial context)
    if 'row' in X.columns and 'col' in X.columns:
        # Create neighborhood statistics (simplified)
        for band in ['B5_nir', 'B6_swir1', 'NDBI', 'NDVI']:
            if band in X.columns:
                X_advanced[f'{band}_neighbor_mean'] = X[band].rolling(window=100, center=True).mean().fillna(X[band].mean())
                X_advanced[f'{band}_neighbor_std'] = X[band].rolling(window=100, center=True).std().fillna(X[band].std())

    # 4. Interaction Features
    interaction_pairs = [
        ('NDBI', 'NDVI'),
        ('B6_swir1', 'B5_nir'),
        ('brightness', 'ratio_swir_nir')
    ]

    for feat1, feat2 in interaction_pairs:
        if feat1 in X.columns and feat2 in X.columns:
            X_advanced[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
            X_advanced[f'{feat1}_div_{feat2}'] = X[feat1] / X[feat2].replace(0, np.nan)

    # 5. Polynomial Features (for non-linear relationships)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    main_features = ['NDBI', 'NDVI', 'B6_swir1', 'B5_nir']
    main_features = [f for f in main_features if f in X.columns]

    if main_features:
        poly_features = poly.fit_transform(X[main_features].fillna(X[main_features].mean()))
        poly_feature_names = poly.get_feature_names_out(main_features)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
        X_advanced = pd.concat([X_advanced, poly_df], axis=1)

    # 6. Log Transformations for skewed features
    skewed_features = ['B6_swir1', 'B5_nir', 'B4_red', 'brightness']
    for feat in skewed_features:
        if feat in X.columns:
            X_advanced[f'log_{feat}'] = np.log1p(X[feat].clip(lower=0))

    return X_advanced

