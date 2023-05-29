# Load best model
load_model = pickle.load(open("xgboost_model.pkl", "rb"))
pred_y = load_model.predict(X_train)
pred_y_test = load_model.predict(X_test)
