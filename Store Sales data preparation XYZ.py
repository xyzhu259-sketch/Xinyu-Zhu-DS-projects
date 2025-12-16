import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

traind = pd.read_csv('C:/Users/angel/Desktop/realds/train_market.csv', parse_dates=['date'])
testd = pd.read_csv('C:/Users/angel/Desktop/realds/test_market.csv', parse_dates=['date'])
oild = pd.read_csv('C:/Users/angel/Desktop/realds/oil.csv', parse_dates=['date'])
stored = pd.read_csv('C:/Users/angel/Desktop/realds/stores.csv')
holidayd = pd.read_csv('C:/Users/angel/Desktop/realds/holidays_events.csv', parse_dates=['date'])


from sklearn.preprocessing import OneHotEncoder
import numpy as np

def prepare_oil_data(oild, maind):
    start_D = min(maind['date'].min(), oild['date'].min())
    end_D = max(maind['date'].max(), oild['date'].max())
    drange = pd.date_range(start_D, end_D, freq = 'D')
    full_dates_oild = pd.DataFrame({'date': drange})
    oild_fin = pd.merge(full_dates_oild, oild, on = 'date', how = 'left')
    oild_fin['dcoilwtico'] = oild_fin['dcoilwtico'].ffill().bfill()
    
    maind = pd.merge(maind, oild_fin[['date', 'dcoilwtico']], on = 'date', how = 'left')
    return maind

testd = prepare_oil_data(oild, testd)
traind = prepare_oil_data(oild, traind)

def prepare_stores(stored, maind):
    maind = pd.merge(maind, stored, on = 'store_nbr', how = 'left')
    return maind
    
traind = prepare_stores(stored, traind)
testd = prepare_stores(stored, testd)

def encode_categories(df):
    state_dummies = pd.get_dummies(df['state'], prefix='state').astype(int)
    type_dummies = pd.get_dummies(df['type'], prefix='type').astype(int)
    city_dummies = pd.get_dummies(df['city'], prefix = 'city').astype(int)
    cluster_dum = pd.get_dummies(df['cluster'], prefix = 'cluster').astype(int)
    family_dummies = pd.get_dummies(df['family'], prefix = 'family').astype(int)
    df = pd.concat([df, state_dummies, type_dummies, city_dummies, cluster_dum], axis=1)
    df = df.drop(['state', 'type','city', 'cluster','family'], axis = 1)
    return df

traind = encode_categories(traind)
testd = encode_categories(testd)

def add_dates(holidayd):
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

holidayd = add_dates(holidayd)

def mergeall(holidayd, df, stored):
    df =  pd.merge(df, stored[['store_nbr', 'city', 'state']], on = 'store_nbr', how = 'left')
    real_holidays = holidayd[holidayd['is_actual_holi'] == True]
    national_holi = real_holidays[real_holidays['is_nationholi'] == 1]
    national_features = national_holi[['date'] +
        [col for col in national_holi.columns if col.startswith('holiday_') or col.startswith('is_')]]
    national_features = national_features.drop_duplicates('date')
    national_features = national_features.add_prefix('national_')
    national_features = national_features.rename(columns={'national_date': 'date'})
    df = pd.merge(df, national_features, on='date', how='left')
    
    regional_holi = real_holidays[real_holidays['is_regionholi'] ==1]
    regional_features = regional_holi[['date', 'locale_name']+
        [col for col in regional_holi.columns if col.startswith('holiday_')]]  
    regional_features = regional_features.add_prefix('regional_')
    regional_features = regional_features.rename(columns = {'regional_date': 'date', 'regional_locale_name': 'state'})
    df = pd.merge(df, regional_features, on = ['date', 'state'], how = 'left')
    
    local_holi = real_holidays[real_holidays['is_localholi'] == 1]
    local_features = local_holi[['date', 'locale_name'] + 
        [col for col in local_holi.columns if col.startswith('holiday_')]]
    local_features = local_features.add_prefix('local_')
    local_features = local_features.rename(columns={
        'local_date': 'date', 
        'local_locale_name': 'city'
    })
    df = pd.merge(df, local_features, on=['date', 'city'], how='left')
    
    holiday_cols = [col for col in df.columns if 'holiday' in col or 'is_' in col]
    for col in holiday_cols:
        if col not in ['date', 'city', 'state']:
            # 尝试转换为数值，无法转换的变为NaN，然后填充0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # 如果该列所有值都是整数，转换为int8以节省内存
            if (df[col] % 1 == 0).all():
                df[col] = df[col].astype('int8')
    
    holiday_type_cols = [col for col in df.columns if 'holiday_type_' in col]
    df['is_holiday'] = df[holiday_type_cols].sum(axis=1) >0
    
    return df

traind = mergeall(holidayd, traind, stored)
testd = mergeall(holidayd, testd, stored)
                                          
def createpayday(df):
    df['is_payday'] = ((df['date'].dt.day==15) | (df['date'].dt.is_month_end)).astype(int)
    return df

traind = createpayday(traind)
testd = createpayday(testd)

def remove_earthquake_by_date(df, earthquake_date='2016-04-16', weeks=4):
    
    earthquake_date = pd.Timestamp(earthquake_date)

    end_date = earthquake_date + pd.Timedelta(weeks=weeks) - pd.Timedelta(days=1)
    
    removed = (df['date'] >= earthquake_date) & (df['date'] <= end_date)

    df_filtered = df[~removed].copy()
    
    return df_filtered

traind = remove_earthquake_by_date(traind, '2016-04-16', weeks=4)

def add_date_features_and_drop(df):

    df = df.copy()
    
    # 添加日期特征
    df['year'] = df['date'].dt.year.astype('int16')
    df['month'] = df['date'].dt.month.astype('int8')
    df['day'] = df['date'].dt.day.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')
    df['quarter'] = df['date'].dt.quarter.astype('int8')
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int8')
    
    # 删除原始的日期列
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    return df

# 在删除city和state列后，添加日期特征并删除日期列
traind = add_date_features_and_drop(traind)
testd = add_date_features_and_drop(testd)

traind = traind.drop(['city', 'state', 'national_holiday_type'], axis=1)
testd = testd.drop(['city', 'state', 'national_holiday_type'], axis=1)


def save_processed_data(traind, testd, output_dir='C:/Users/angel/Desktop/realds/processed/'):

    import os
    os.makedirs(output_dir, exist_ok=True)
    
    traind_file = os.path.join(output_dir, 'train_processed.csv')
    testd_file = os.path.join(output_dir, 'test_processed.csv')
    
    traind.to_csv(traind_file, index=False)
    testd.to_csv(testd_file, index=False)
    
    print(f"data saved to: {output_dir}")

    return traind_file, testd_file

save_processed_data(traind, testd)

