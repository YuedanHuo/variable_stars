import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from scipy.fft import fft
import lightgbm as lgb


def fold_time_series(time_point, period, div_period):
    return (time_point -
            (time_point // (period / div_period)) * period / div_period)

def get_bin_means(X_df, num_bins, band):
    feature_array = np.empty((len(X_df), num_bins))

    for k, (_, x) in enumerate(X_df.iterrows()):
        period = x['period']
        div_period = x['div_period']
        real_period = period / div_period
        bins = [i * real_period / num_bins for i in range(num_bins + 1)]

        time_points = np.array(x['time_points_' + band])
        light_points = np.array(x['light_points_' + band])
        time_points_folded = \
            np.array([fold_time_series(time_point, period, div_period)
                      for time_point in time_points])
        time_points_folded_digitized = \
            np.digitize(time_points_folded, bins) - 1

        for i in range(num_bins):
            this_light_points = light_points[time_points_folded_digitized == i]
            if len(this_light_points) > 0:
                feature_array[k, i] = np.mean(this_light_points)
            else:
                feature_array[k, i] = np.nan  # missing

    return feature_array

def fourier_features(X_df, num_components, band):
    feature_array = np.empty((len(X_df), num_components))

    for k, (_, x) in enumerate(X_df.iterrows()):
        light_points = np.array(x['light_points_' + band])
        transformed = fft(light_points)
        frequencies = np.abs(transformed)[:num_components]
        feature_array[k,:] = frequencies

    return feature_array


num_bins = 5
transformer_r = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, num_bins, 'r')
)

transformer_b = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, num_bins, 'b')
)

num_components = 5
transformer_f_r = FunctionTransformer(
    lambda X_df: fourier_features(X_df, num_components, 'r')
)
transformer_f_b = FunctionTransformer(
    lambda X_df: fourier_features(X_df, num_components, 'b')
)

cols = [
    'magnitude_b',
    'magnitude_r',
    'period',
    'asym_b',
    'asym_r',
    'log_p_not_variable',
    'sigma_flux_b',
    'sigma_flux_r',
    'quality',
    'div_period',
]

common = ['period', 'div_period']
transformer = make_column_transformer(
    (transformer_r, common + ['time_points_r', 'light_points_r']),
    (transformer_b, common + ['time_points_b', 'light_points_b']),
    (transformer_f_r, ['light_points_r']),
    (transformer_f_b, ['light_points_b']),
    ('passthrough', cols)
)

# Example model pipeline using LightGBM
model_pipeline = make_pipeline(
    transformer,
    SimpleImputer(strategy='most_frequent'),
    lgb.LGBMClassifier(
        max_depth=10,
        n_estimators=200,
        learning_rate=0.05,
        min_child_samples=60,
        subsample=0.63,
        colsample_bytree=0.8,
        boosting_type='gbdt'
    )
)

def get_estimator():
    return model_pipeline

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from optuna import samplers
import problem

X_df, y = problem.get_train_data()
# Transform data and handle missing values
X_transform = transformer.fit_transform(X_df)
imputer = SimpleImputer()
X_fillna = imputer.fit_transform(X_transform)

import optuna
import lightgbm as lgb

def objective(trial):
    trial.suggest_int('early_stopping_rounds', 10, 50)

    # Hyperparameter search space
    max_depth = trial.suggest_int('max_depth', 3, 12)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    min_child_samples = trial.suggest_int('min_child_samples', 10, 100)
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    
    # Initialize LightGBM model
    reg = lgb.LGBMClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        boosting_type=boosting_type,
        random_state=42,
        verbosity=-1
    )
    
    # Cross-validation evaluation (using accuracy score as metric)
    score = np.mean(cross_val_score(reg, X_fillna, y, cv=5, n_jobs=-1, scoring="accuracy"))
    
    return score

# Setup Optuna study and optimization
sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler, direction='maximize')
optuna.logging.disable_default_handler()  # limit verbosity

# Start the optimization process
study.optimize(objective, n_trials=50)

# Show best result
print("Best Trial Parameters:", study.best_trial.params)
print("Best Trial Accuracy:", study.best_trial.value)

