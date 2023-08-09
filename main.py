import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial
import pandas as pd
from ramda import pipe
from functools import partial
import numpy as np
min = 5
max = 7
minbetodd = 1.9
maxbetodd = 3.5
insufficient = 20
dif_threshold = 1
from common import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, addSumStats, dropInsufficient, dif_thresholdd

import pandas as pd





df = pd.read_csv('/home/kk/PycharmProjects/oddmaker/data/exp23_withLeagues_LIMITED_minutes_4-35_odd_1.1-5_insfufficient_10_dif_1_onlyDraws.csv')
df.dropna(inplace=True)
print(df.head(5))
print(df.shape)


pipeline = pipe(
    removeDotFromColumnNames,
    dropMinutes,
)
# Apply pipeline
df = pipeline(df)
print(df.shape)