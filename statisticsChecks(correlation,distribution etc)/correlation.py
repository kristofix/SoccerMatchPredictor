import pandas as pd
from ramda import pipe
import seaborn as sns
import matplotlib.pyplot as plt
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
# df = df.iloc[:len(df)//100]      #uncomment for quick test run


correlation_matrix = df.corr()
print(correlation_matrix)

#Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap")
plt.show()

#Correlation threshold
threshold = 0.8

# Identify highly correlated features
high_corr_vars = []
for col in correlation_matrix.columns:
    for idx in correlation_matrix.index:
        if idx == col:
            #self-correlation
            continue
        if abs(correlation_matrix.loc[idx, col]) > threshold:
            high_corr_vars.append((idx, col))

print("Highly Correlated Feature Pairs(over ",threshold,"): ")
for var1, var2 in high_corr_vars:
    print(f"{var1} and {var2}")
