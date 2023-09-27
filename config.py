# Models settings
n_iter = 10 # BayesSearch for lgbm,xgb and catboost
cv = 5 # BayesSearch for lgbm,xgb and catboost
epochs = 200 # for neuralnetwork
patience = 52 # for neuralnetwork

# filter soccer dataset
min_time = 19
max_time = 21
minbetodd = 1.9
maxbetodd = 5
insufficient = 20 # minimum amount of information from each sports meeting
threshold = 5 # minimum difference between two teams summary stats