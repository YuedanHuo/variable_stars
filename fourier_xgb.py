import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.fft import fft
from sklearn.ensemble import HistGradientBoostingClassifier


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

# Example model pipeline
model_pipeline = make_pipeline(
    transformer,
    SimpleImputer(strategy='most_frequent'),
    HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.13912546584820343,
        l2_regularization=1.7158589997464462e-05,
        min_samples_leaf=60,
        max_iter=1000
    )
)

def get_estimator():
    return model_pipeline

# Cross-validation
import problem
X_df, y = problem.get_train_data()
y -= 1

# cross validation
#skf = StratifiedKFold(n_splits=5)
#scores = []
#for train_idx, test_idx in skf.split(X_df, y):
#    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
#    y_train, y_test = y[train_idx], y[test_idx]
    
#    model_pipeline.fit(X_train, y_train)
#    y_pred = model_pipeline.predict(X_test)
#    score = accuracy_score(y_test, y_pred)
#    scores.append(score)
#print("Cross-validation accuracy:", np.mean(scores))


import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from optuna import samplers
X_transform = transformer.fit_transform(X_df)
imputer = SimpleImputer()
X_fillna = imputer.fit_transform(X_transform)

def objective(trial):
    # Train-test split
    #X_train, X_val, y_train, y_val = train_test_split(X_fillna, y, test_size=0.2, random_state=42)
    trial.suggest_int('early_stopping_rounds', 10, 50)
    
    # Define hyperparameters to tune
    max_depth = trial.suggest_int('max_depth', 2, 32)
    learning_rate = trial.suggest_float('learning_rate', 10**-5, 10**0, log=True)
    l2_regularization = trial.suggest_float('l2_regularization', 10**-5, 10**0, log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)

    # Initialize regressor
    reg = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        max_iter=1000,
        random_state=42
    )
    

    # Cross-validation evaluation (using R2 score as metric)
    score = np.mean(cross_val_score(reg, X_fillna, y, cv=5, n_jobs=-1, scoring="accuracy"))
    
    return score

# Setup Optuna study and optimization
sampler = samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler, direction='maximize')
optuna.logging.disable_default_handler()  # limit verbosity



# Start the optimization process
study.optimize(objective, n_trials=50)

# Show best result
print("Best Trial Parameters:", study.best_trial.params)
print("Best Trial R2 Score:", study.best_trial.value)
