# Models settings
n_iter = 2 # BayesSearch for lgbm,xgb and catboost
cv = 5 # BayesSearch for lgbm,xgb and catboost
epochs = 200 # for neuralnetwork
patience = 20 # for neuralnetwork

# filter soccer dataset
min_time = 14
max_time = 16
minbetodd = 2
maxbetodd = 10
insufficient = 20 # minimum amount of information from each sports meeting
threshold = 1 # minimum difference between two teams summary stats