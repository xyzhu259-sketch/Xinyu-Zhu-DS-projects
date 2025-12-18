import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

traind = pd.read_csv('C:/Users/angel/Desktop/realds/Store Sales Raw Data/train.csv', parse_dates=['date'])
testd = pd.read_csv('C:/Users/angel/Desktop/realds/Store Sales Raw Data/test.csv', parse_dates=['date'])
oild = pd.read_csv('C:/Users/angel/Desktop/realds/Store Sales Raw Data/oil.csv', parse_dates=['date'])
stored = pd.read_csv('C:/Users/angel/Desktop/realds/Store Sales Raw Data/stores.csv')
holidayd = pd.read_csv('C:/Users/angel/Desktop/realds/Store Sales Raw Data/holidays_events.csv', parse_dates=['date'])


from sklearn.preprocessing import OneHotEncoder
import numpy as np

def prepare_oil_data(oild, maind):
    start_D = min(maind['date'].min(), oild['date'].min())
    end_D = max(maind['date'].max(), oild['date'].max())
    drange = pd.date_range(start_D, end_D, freq='D')
    full_dates_oild = pd.DataFrame({'date': drange})
    oild_fin = pd.merge(full_dates_oild, oild, on='date', how='left')
    oild_fin['dcoilwtico'] = oild_fin['dcoilwtico'].ffill().bfill()
    
    # add inverse of oil price, which can be positively related to store sales
    oild_fin['inv_oil'] = 1 / oild_fin['dcoilwtico']
    
    # add rate of change in oil price, as oil price is shown to be a large determinant in sales
    oild_fin['oil_change'] = oild_fin['dcoilwtico'].pct_change()
    oild_fin['oil_change'] = oild_fin['oil_change'].fillna(0)
    
    maind = pd.merge(maind, oild_fin[['date', 'dcoilwtico', 'inv_oil', 'oil_change']], 
                     on='date', how='left')
    
    return maind

testd = prepare_oil_data(oild, testd)
traind = prepare_oil_data(oild, traind)

def prepare_stores(stored, maind):
    maind = pd.merge(maind, stored, on = 'store_nbr', how = 'left')
    return maind
    
traind = prepare_stores(stored, traind)
testd = prepare_stores(stored, testd)

#grouping stores according to mean sales, enlarging the effect of store sales experience
def create_store_bands(train, test, n_bands=6):

    store_avg_sales = train.groupby('store_nbr')['sales'].mean()
    
    store_bands = pd.qcut(store_avg_sales, q=n_bands, labels=range(1, n_bands+1))
    store_band_dict = store_bands.to_dict()
    
    train['store_band'] = train['store_nbr'].map(store_band_dict)
    test['store_band'] = test['store_nbr'].map(store_band_dict)
    
    band_stats = train.groupby('store_band')['sales'].agg(['mean', 'std', 'median']).round(2)
    band_stats.columns = [f'band_{col}' for col in band_stats.columns]
    band_stats = band_stats.reset_index()
    
    train = pd.merge(train, band_stats, on='store_band', how='left')
    test = pd.merge(test, band_stats, on='store_band', how='left')
    
    for col in ['band_mean', 'band_std', 'band_median']:
        if col in test.columns:
            global_mean = train['sales'].mean()
            test[col] = test[col].fillna(global_mean)
    
    return train, test

traind, testd = create_store_bands(traind, testd, n_bands=6)

def encode_categories(df, use_store_band=True):
    df = df.copy()

    # creating label encoding for the family category
    if 'family' in df.columns:
        fam_map = {family: i for i, family in enumerate(df['family'].unique())}
        df['family'] = df['family'].map(fam_map).astype('int8')
    
    # one-hot coding for store types: decrease possible linear relationship.
    if 'type' in df.columns:
        type_dummies = pd.get_dummies(df['type'], prefix='type', dtype='int8')
        df = pd.concat([df, type_dummies], axis=1)
        df = df.drop(['type'], axis=1)
    
    # deal with store type outliers: rare types to be considered together
    if 'type' not in df.columns:
        for col in ['type_A', 'type_B', 'type_C', 'type_D', 'type_E']:
            if col not in df.columns:
                df[col] = 0
    
    return df

traind = encode_categories(traind)
testd = encode_categories(testd)

def add_holidates(holidayd):
    transfers = holidayd[holidayd['transferred'] == True]
    transfertos = holidayd[holidayd['type'] == 'Transfer']
    holidayd['is_actual_holi'] = False
    holidayd['is_original_holi'] = False
    
    transfer_dates = transfers['date'].tolist()
    holidayd.loc[holidayd['date'].isin(transfer_dates), 'is_actual_holi'] = False
    holidayd.loc[holidayd['date'].isin(transfer_dates), 'is_original_holi'] = True
    
    transferto_dates = transfertos['date'].tolist()
    holidayd.loc[holidayd['date'].isin(transferto_dates), 'is_actual_holi'] = True
    holidayd.loc[holidayd['date'].isin(transferto_dates), 'type'] = 'Holiday'
    other_holidays_mask = (holidayd['type'].isin(['Bridge', 'Additional', 'Holiday'])\
                           & holidayd['transferred'] == False &\
                               (~holidayd['date'].isin(transfers)))
    holidayd.loc[other_holidays_mask, 'is_actual_holi'] = True

    real_holi = holidayd[holidayd['is_actual_holi'] == True]
    holiday_feat = pd.DataFrame({
        'date': holidayd['date'],
        'holiday_type': holidayd['type'],
        'locale': holidayd['locale'],
        'locale_name': holidayd['locale_name'],
        'is_actual_holi': holidayd['is_actual_holi'] })
    type_dummies = pd.get_dummies(holiday_feat['holiday_type'], prefix='holiday_type')
    holidayd = pd.concat([holiday_feat, type_dummies], axis=1)

    locale_dummies = pd.get_dummies(holidayd['locale'], prefix='holiday_locale')
    holidayd = pd.concat([holidayd, locale_dummies], axis=1)
    holidayd['is_nationholi'] = (holidayd['locale'] == 'National').astype(int)
    holidayd['is_regionholi'] = (holidayd['locale'] == 'Regional').astype(int)
    holidayd['is_localholi'] = (holidayd['locale'] == 'Local').astype(int)
    return holidayd

holidayd = add_holidates(holidayd)

#Based on Kaggle discussions, some stores seem to be outliers, distracting the sales pattern for model training.
def remove_outlier_stores(df, outlier_stores=None):

    outlier_stores = [45, 44, 1, 2, 3, 4, 5, 9, 7, 12, 20]
    
    return df[~df['store_nbr'].isin(outlier_stores)].copy()

traind = remove_outlier_stores(traind)


def mergeall(holidayd, df, stored):
    
    if 'state' not in df.columns or 'city' not in df.columns:
        store_info = stored[['store_nbr', 'city', 'state']].drop_duplicates()
        df = pd.merge(df, store_info, on='store_nbr', how='left', suffixes=('', '_store'))
    
    # acquire real holidays
    real_holidays = holidayd[holidayd['is_actual_holi'] == True].copy()
    
    #adjust for different types of holidays
    national_data = real_holidays[real_holidays['is_nationholi'] == 1]['date'].unique()
    df['is_national_holiday'] = df['date'].isin(national_data).astype('int8')
    
    regional_data = real_holidays[real_holidays['is_regionholi'] == 1][['date', 'locale_name']].copy()
    regional_data = regional_data.rename(columns={'locale_name': 'state'})
    
    regional_data['is_regional_holiday'] = 1
    regional_data = regional_data.drop_duplicates(['date', 'state'])
    
    df = pd.merge(df, regional_data, on=['date', 'state'], how='left')
    df['is_regional_holiday'] = df['is_regional_holiday'].fillna(0).astype('int8')
    
    local_data = real_holidays[real_holidays['is_localholi'] == 1][['date', 'locale_name']].copy()
    local_data = local_data.rename(columns={'locale_name': 'city'})
    
    local_data['is_local_holiday'] = 1
    local_data = local_data.drop_duplicates(['date', 'city'])
    
    df = pd.merge(df, local_data, on=['date', 'city'], how='left')
    df['is_local_holiday'] = df['is_local_holiday'].fillna(0).astype('int8')
    
    df['is_holiday'] = ((df['is_national_holiday'] == 1) | 
                        (df['is_regional_holiday'] == 1) | 
                        (df['is_local_holiday'] == 1)).astype('int8')
    
    return df
                  
traind = mergeall(holidayd, traind, stored)
testd = mergeall(holidayd, testd, stored)
                        
def createpayday(df):
    df['is_payday'] = ((df['date'].dt.day==15) | (df['date'].dt.is_month_end)).astype(int)
    return df

traind = createpayday(traind)
testd = createpayday(testd)

# remove earthquake affected data from the dataset
def remove_earthquake_by_date(df, earthquake_date='2016-04-16', weeks=4):
    
    earthquake_date = pd.Timestamp(earthquake_date)

    end_date = earthquake_date + pd.Timedelta(weeks=weeks) - pd.Timedelta(days=1)
    
    removed = (df['date'] >= earthquake_date) & (df['date'] <= end_date)

    df_filtered = df[~removed].copy()
    
    return df_filtered

traind = remove_earthquake_by_date(traind, '2016-04-16', weeks=4)

# adding date data, including the day, month, season, year, is or not weekend, and whether it is the start or end of a year, month.
def add_dates(df):

    df = df.copy()
    
    # adjust date labels
    df['year'] = df['date'].dt.year.astype('int16')
    df['month'] = df['date'].dt.month.astype('int8')
    df['day'] = df['date'].dt.day.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int8')
    df['quarter'] = df['date'].dt.quarter.astype('int8')
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int8')
    df['is_month_start'] = df['date'].dt.is_month_start.astype('int8')
    df['is_month_end'] = df['date'].dt.is_month_end.astype('int8')
    df['is_year_start'] = df['date'].dt.is_year_start.astype('int8')
    df['is_year_end'] = df['date'].dt.is_year_end.astype('int8')
    
    df['season'] = pd.cut(df['month'], 
                          bins=[0, 3, 6, 9, 12, 13],
                          labels=[2, 3, 4, 1, 2],
                          ordered=False).astype('int8')  # 1,2,3,4 for seasons spring to winter
    
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    return df

traind = add_dates(traind)
testd = add_dates(testd)

traind = traind.drop(['city', 'state'], axis=1)
testd = testd.drop(['city', 'state'], axis=1)

# saving data to an intermediate csv file
def save_data(traind, testd, output_dir='C:/Users/angel/Desktop/realds/processed/'):

    import os
    os.makedirs(output_dir, exist_ok=True)
    
    traind_file = os.path.join(output_dir, 'train_processed.csv')
    testd_file = os.path.join(output_dir, 'test_processed.csv')
    
    traind.to_csv(traind_file, index=False)
    testd.to_csv(testd_file, index=False)
    
    print(f"data saved to: {output_dir}")

    return traind_file, testd_file

save_data(traind, testd)

