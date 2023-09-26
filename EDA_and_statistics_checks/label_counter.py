from data_preparation import data_preparation
import matplotlib.pyplot as plt

df = data_preparation()

print(df['zzz_play'].value_counts())
print(df['zzz_play'].value_counts(normalize=True) * 100)
df['zzz_play'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()


