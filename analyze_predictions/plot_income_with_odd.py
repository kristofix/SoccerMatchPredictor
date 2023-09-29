import pandas as pd
import matplotlib.pyplot as plt

def plot_income_flow():
    df = pd.read_csv('/home/kk/PycharmProjects/oddmaker/test_and_predictions.csv')
    def determine_odd_winner(row):
        if row['zzz_play'] == 2.0:
            return row['framesawayodd']
        elif row['zzz_play'] == 1.0:
            return row['frameshomeodd']
        elif row['zzz_play'] == 0.0:
            return 0

    # Apply the function row-wise
    df['odd_winner'] = df.apply(determine_odd_winner, axis=1)

    df['profit'] = df.apply(lambda row: row['odd_winner'] - 1 if row['zzz_play'] == row['predict'] else -1, axis=1)

    df['cumulative_profit'] = df['profit'].cumsum()

    plt.figure(figsize=(10,6))
    plt.plot(df['cumulative_profit'])
    plt.title('Cumulative Profit Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit')
    plt.grid(True)

    print(df.head(5))

    df.to_csv('test_and_predictions_with_profit.csv', index=False)

    plt.show()