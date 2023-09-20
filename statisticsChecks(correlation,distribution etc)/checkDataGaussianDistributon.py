import pandas as pd
from ramda import pipe
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib
# matplotlib.use('TkAgg')
from commonFunction import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_threshold

df = pd.read_csv('../data/exp23_withLeagues_LIMITED_minutes_4-35_odd_1.1-5_insfufficient_10_dif_1_onlyDraws.csv')
df.dropna(inplace=True)
print(df.shape)
pipeline = pipe(
    removeDotFromColumnNames,
    dropMinutes,
    sortByDate,
    dropNotDraw,
    oddsFilter,
    addSumStats,
    dropInsufficient,
    dif_threshold
)

df = pipeline(df)
df.drop(['datetimestamp'], axis=1, inplace=True)

#Stratified sampling ensures that each class is represented proportionally. Need to use it because shapiro-wilk can't handle big dataset
from sklearn.model_selection import train_test_split
_, sample_df = train_test_split(df, test_size=0.01, stratify=df['zzz_play'])

#Shapiro-Wilk test
for col in sample_df.columns[:-1]:
    _, p_value = stats.shapiro(sample_df[col])
    print(p_value)
    if p_value > 0.05:
        print(f"{col} likely follows a Gaussian distribution!!!!!!!")
    else:
         print(f"{col} does not follow a Gaussian distribution.")

#Take a look on each column histogram
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for col in sample_df.columns[:-1]:
    sample_df[col].plot(kind='hist', bins=20, alpha=0.5, label=col, ax=ax)
ax.legend()
plt.show()





