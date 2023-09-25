from scipy import stats
import numpy as np

from data_preparation import data_preparation

df = data_preparation()
# df = df.head(1000)      #uncomment for quick test run

def addSumStats(df):
    # Compute new columns and hold them temporarily
    sumABstats = df['frameshomeshotsOnTarget'] + df['frameshomeshotsOffTarget'] + df['frameshomeattacks'] + df['frameshomedangerousAttacks'] + df['framesawayshotsOnTarget'] + df['framesawayshotsOffTarget'] + df['framesawayattacks'] + df['framesawaydangerousAttacks']
    # Insert new columns before the last column
    df.insert(loc=len(df.columns) - 1, column='sumABstats', value=sumABstats)
    return df

df = addSumStats(df)

print(df.columns)

group1 = df.loc[df['zzz_play'] == 0]
group2 = df.loc[df['zzz_play'] != 0]

# Obliczenie t-testu dla dwóch prób
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"Statystyka t: {t_stat}")
print(f"Wartość p: {p_value}")


# Obliczenie średnich dla group1 i group2
mean_group1 = np.mean(group1)
mean_group2 = np.mean(group2)

print(f"Średnia dla group1: {mean_group1}")
print(f"Średnia dla group2: {mean_group2}")

# Interpretacja wyników
alpha = 0.05
if p_value < alpha:
    print("Odrzucamy hipotezę zerową, średnie są różne.")
else:
    print("Nie odrzucamy hipotezy zerowej, nie ma wystarczających dowodów na to, że średnie są różne.")

print(df.info())

print(group1.info())
print(group1.head())
print(group2.info())
print(group2.head())