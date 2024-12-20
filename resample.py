import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

# Define the phase grid
t = np.linspace(0, 1, 100)

# Fix the resample function
def resample(star):
    t_i = (star['time_points_b'] % star['period']) / star['period']
    t_r = (star['time_points_r'] % star['period']) / star['period']
    
    # Interpolate the light points instead of time points
    interp_b = interpolate.interp1d(
        t_i, star['light_points_b'],
        kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    interp_r = interpolate.interp1d(
        t_r, star['light_points_r'],
        kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    
    # Combine the two resampled signals into a single array
    resampled = np.column_stack([interp_b(t), interp_r(t)])
    return resampled

# Apply resampling to the entire dataset
def resample_dataset(X):
    return np.array([resample(row) for _, row in X.iterrows()])

# Pipeline with resampling and time-series classification
clf = make_pipeline(
    FunctionTransformer(resample_dataset, validate=False),
    KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw", n_jobs=-1)
)


from sklearn.model_selection import cross_val_score, StratifiedKFold
import problem

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


X, y = problem.get_train_data()
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

# Print the results
print(f"Cross-Validation Scores: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation: {np.std(scores):.4f}")

