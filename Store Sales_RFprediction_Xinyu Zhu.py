import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

traind = pd.read_csv('C:/Users/angel/Desktop/realds/processed/train_processed.csv')
testd = pd.read_csv('C:/Users/angel/Desktop/realds/processed/test_processed.csv')

X_train = traind.drop('sales', axis=1)
y_train = traind['sales']
X_test = testd

feature_names = X_train.columns.tolist()

X_test = X_test[feature_names]

#Model setup
rf_model = RandomForestRegressor(
    n_estimators=800,
    max_depth=16,
    min_samples_split=10, 
    min_samples_leaf=4,
    random_state=1,
    n_jobs=-1
)

print("training model: ")
rf_model.fit(X_train, y_train)

# 评估
train_pred = rf_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2 = r2_score(y_train, train_pred)

print(f"train set RMSE: {train_rmse:.4f}")
print(f"train set R²: {train_r2:.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("top 10 features: ")
print(feature_importance.head(10))

predictions = rf_model.predict(X_test)

# creating file submission
original_test = pd.read_csv('C:/Users/angel/Desktop/realds/test_market.csv')
submission = pd.DataFrame({
    'id': original_test['id'],
    'sales': predictions
})

submission.to_csv('C:/Users/angel/Desktop/realds/rf_predictions.csv', index=False)
print("done")
