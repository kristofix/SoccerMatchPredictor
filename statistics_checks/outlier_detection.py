from scipy import stats
import numpy as np

def z_score_outlier(df):
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df))

    print('Amount of data before outlier detection:', len(df))

    # Remove outliers
    df_no_outliers = df[(z_scores <= 3).all(axis=1)] # About 99.7% falls within three standard deviations (Z-score between -3 and 3)

    print('Amount of data after outlier detection:', len(df_no_outliers))
    print('Number of outliers removed:', len(df) - len(df_no_outliers))

    return df_no_outliers

