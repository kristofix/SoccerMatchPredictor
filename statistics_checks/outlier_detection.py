from scipy import stats
from data_preparation import data_preparation
import numpy as np

def z_score_outlier(df):
    df = data_preparation()
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df))
    outliers = (z_scores > 3) # About 99.7% falls within three standard deviations (Z-score between -3 and 3)

    return df[(z_scores <= 3).all(axis=1)]