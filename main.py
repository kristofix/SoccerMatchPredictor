import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from functools import partial
import pandas as pd
from ramda import pipe
from functools import partial
import numpy as np
import wandb
from wandb.keras import WandbCallback
from wandb.xgboost import WandbCallback
#wandb login
from config import min_time, max_time, minbetodd, maxbetodd, insufficient, dif_threshold
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_thresholdd

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
    dif_thresholdd
)

# Apply pipeline
df = pipeline(df)
print(df.shape)
#print(df.head(10))


