import seaborn as sns
import matplotlib.pyplot as plt
from data_preparation import data_preparation

df = data_preparation()
# df = df.head(1000)      #uncomment for quick test run

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
