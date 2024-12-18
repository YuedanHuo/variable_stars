import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models


# Function to fold time series
def fold_time_series(time_point, period, div_period):
    return (time_point -
            (time_point // (period / div_period)) * period / div_period)


# Function to get bin means from time series
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


# Define the autoencoder
def define_autoenc(X_input):
    input_layer = layers.Input(shape=(X_input.shape[1],))
    x = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(X_input.shape[1], activation='linear')(x)

    autoencoder = models.Model(inputs=input_layer, outputs=decoded)  # Reconstructed output
    encoder = models.Model(inputs=input_layer, outputs=encoded)  # Encoder for feature extraction
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


# Custom transformer to apply the encoder for feature extraction
class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder_model = None
    
    def fit(self, X, y=None):
        # Train the autoencoder during fitting
        self.encoder_model = define_autoenc(X)
        self.encoder_model.fit(X, X, epochs=50, batch_size=32, verbose=0)
        return self
    
    def transform(self, X):
        # Use the trained encoder to extract compressed features
        return self.encoder_model.predict(X)  # Extract compressed features


# Function to create the full pipeline
def get_estimator():
    # Example of how to apply bin means transformation
    transformer_r = FunctionTransformer(
        lambda X_df: get_bin_means(X_df, 5, 'r')
    )
    
    transformer_b = FunctionTransformer(
        lambda X_df: get_bin_means(X_df, 5, 'b')
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
        ('passthrough', cols)
    )

    # Build the pipeline
    pipe = make_pipeline(
        transformer,  # Apply bin mean transformation first
        SimpleImputer(strategy='most_frequent'),
        AutoencoderTransformer(),  # Use the autoencoder transformer
        XGBClassifier(max_depth=5, n_estimators=10, use_label_encoder=False, eval_metric='logloss')
    )

    return pipe
