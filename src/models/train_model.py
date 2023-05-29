import xgboost as xgb
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from make_dataset import *

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],
    'scale_pos_weight': [1, 5, 10]
}

# Create an instance of the XGBoost Classifier
xgb_model = xgb.XGBClassifier()

# Create a KFold object with the desired number of folds (k)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scorer = make_scorer(f1_score)

# Perform grid search using GridSearchCV with k-fold cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scorer, cv=kfold)

# Fit the grid search object to your training data
grid_search.fit(X_train, y_train)

# Access the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

# Save trained best model
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
