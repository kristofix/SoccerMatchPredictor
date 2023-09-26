from scipy import stats
from data_preparation import data_preparation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = data_preparation()

#Stratified sampling ensures that each class is represented proportionally. Need to use it because shapiro-wilk can't handle big dataset
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
fig, ax = plt.subplots()
for col in sample_df.columns[:-1]:
    sample_df[col].plot(kind='hist', bins=20, alpha=0.5, label=col, ax=ax)
ax.legend()
plt.show()





