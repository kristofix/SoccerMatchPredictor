import pandas as pd
from ramda import pipe
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib
# matplotlib.use('TkAgg')
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_threshold

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

# Apply pipeline
df = pipeline(df)
df.drop(['datetimestamp'], axis=1, inplace=True)
print(df.shape)
df = df.iloc[:len(df)//100]      #uncomment for quick test run
print(df.shape)

correlation_matrix = df.corr()

# Wypisanie macierzy korelacji
print("Correlation matrix:")
print(correlation_matrix)

#Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa ciepła korelacji")
plt.show()

#Correlation threshold
threshold = 0.8

# Identify highly correlated features
high_corr_vars = []
for col in correlation_matrix.columns:
    for idx in correlation_matrix.index:
        if idx == col:
            # Skip self-correlation
            continue
        if abs(correlation_matrix.loc[idx, col]) > threshold:
            # Record the high correlation pair
            high_corr_vars.append((idx, col))

# Remove duplicates
high_corr_vars = list(set([tuple(sorted(pair)) for pair in high_corr_vars]))
print(high_corr_vars)
# Print out highly correlated feature pairs
print("Highly Correlated Feature Pairs:")
for var1, var2 in high_corr_vars:
    print(f"{var1} and {var2}")
