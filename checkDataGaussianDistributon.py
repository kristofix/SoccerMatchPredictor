import pandas as pd
from ramda import pipe
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib
# matplotlib.use('TkAgg')
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_threshold

df = pd.read_csv('/home/kk/PycharmProjects/oddmaker/data/exp23_withLeagues_LIMITED_minutes_4-35_odd_1.1-5_insfufficient_10_dif_1_onlyDraws.csv')
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

# Apply pipeline
df = pipeline(df)
df.drop(['datetimestamp'], axis=1, inplace=True)
print(df.shape)
#df = df.iloc[:len(df)//100]      #uncomment for quick test run
print(df.shape)

#Histogram plots
fig, axes = plt.subplots(3, 5, figsize=(15, 10))  # Adjust grid size (3 rows, 5 columns)
axes = axes.ravel()
for i, col in enumerate(df.columns[:-1]):  # Assuming last column is the target
    axes[i].hist(df[col], bins=20, color='blue', alpha=0.7)
    axes[i].set_title(col)
plt.tight_layout()
plt.show()


#Shapiro-Wilk test
for col in df.columns[:-1]:  # Assuming last column is target
    _, p_value = stats.shapiro(df[col])
    if p_value > 0.05:
        print(f"{col} likely follows a Gaussian distribution.")
    else:
         print(f"{col} does not follow a Gaussian distribution.")


#Quantile-Quantile test
n_cols = len(df.columns[:-1])
n_rows = int((n_cols + 1) / 2)  # Assuming two columns per row
fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows*4))
flat_axes = axes.flatten()
for idx, col in enumerate(df.columns[:-1]):  # Assuming last column is target
    _, p_value = stats.shapiro(df[col])
    if p_value > 0.05:
        print(f"{col} likely follows a Gaussian distribution.")
    else:
        print(f"{col} does not follow a Gaussian distribution.")

    # Generate Q-Q plot for the column
    stats.probplot(df[col], plot=flat_axes[idx])
    flat_axes[idx].set_title(f'Q-Q plot for {col}')

plt.tight_layout()
plt.show()
