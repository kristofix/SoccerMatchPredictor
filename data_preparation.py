import pandas as pd
from ramda import pipe
from common_function import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter,addSumStats, dif_threshold, dropInsufficient, dropUnnecessary

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
        dropUnnecessary
    )

    return pipeline(df)
