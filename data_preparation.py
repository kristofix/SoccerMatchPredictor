import pandas as pd
from ramda import pipe
from common_function import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter,addSumStats, dif_threshold, dropInsufficient, dropUnnecessary, normalize_data, standarize_data
from EDA_and_statistics_checks.outlier_detection import z_score_outlier
def data_preparation():

    df = pd.read_csv('/home/kk/PycharmProjects/oddmaker/data/exp23_withLeagues_LIMITED_minutes_4-35_odd_1.1-5_insfufficient_10_dif_1_onlyDraws.csv')
    df.dropna(inplace=True)

    pipeline = pipe(
        removeDotFromColumnNames,
        dropMinutes,
        sortByDate,
        dropNotDraw,
        oddsFilter,
        addSumStats,
        dropInsufficient,
        dif_threshold,
        dropUnnecessary,
        z_score_outlier,
        # normalize_data,
        # standarize_data
    )

    return pipeline(df)
