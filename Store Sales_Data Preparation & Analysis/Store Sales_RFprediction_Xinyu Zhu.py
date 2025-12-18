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

# model setup: train different models for each store band
def train_grouped_models(train_df, test_df):
    
    models = {}
    all_predictions = pd.Series(index=test_df.index, dtype=float)
    
    groups = train_df.groupby(['family', 'store_band'])
    
    for (family, store_band), group in groups:
        if len(group) >= 50: 
            X_train = group.drop('sales', axis=1)
            y_train = group['sales']
            
            model = RandomForestRegressor(
                n_estimators=250,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=1,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            key = f"{family}_{store_band}"
            models[key] = model
            
            test_mask = (test_df['family'] == family) & (test_df['store_band'] == store_band)
            test_subset = test_df[test_mask]
            
            if len(test_subset) > 0:
                preds = model.predict(test_subset)
                all_predictions.loc[test_subset.index] = preds
                
    print(f"trained: {len(models)} grouped models.")
    
    
    return all_predictions, models

predictions, models = train_grouped_models(traind, testd)


# creating file submission
original_test = pd.read_csv('C:/Users/angel/Desktop/realds/test_market.csv')
submission = pd.DataFrame({
    'id': original_test['id'],
    'sales': predictions
})

submission.to_csv('C:/Users/angel/Desktop/realds/rf_predictions.csv', index=False)
print("done")
