Store Sales - Time Series Forecasting 
-- Data analysis and prediction using Random Forest Regression by Xinyu Zhu

Project Description:
  The Store Sales competition is a competitive project from Kaggle, aiming to predict the Ecuadorian store Favorita's sales using given past data regarding goods information, time and oil price. The project involves data prediction with time & date related
data, feature engineering and the use of machine learning algorithms.\

Contest information:
  Name: Store Sales - Time Series Forecasting
  Platform: Kaggle
  Contest type: Regression forecasting
  Evaluation criterion: RMSLE

Project Structure:
  project/
  │
  ├── Store Sales Raw Data/                          # raw data file
  │   ├── train.csv                  # train data
  │   ├── test.csv                   # test data
  │   ├── oil.csv                    # oil price for different dates
  │   ├── stores.csv                 # store information
  │   └── holidays_events.csv        # holiday date information
  │   └── transactions.csv        # transaction data
  │
  ├── Store Sales_Data Preparation & Analysis/        # codes of data processing & analysis
  │   ├── Store Sales data preparation_Xinyu Zhu.py        # data processing
  │   ├── Store Sales_RFprediction_Xinyu Zhu.py          #model training
  │
  ├── final_predictions.csv
  │
  ├── requirements.txt               # 依赖包列表
  ├── README.md                      # 项目说明
  └── main.py                        # 主程序入口

Technology Stack:
  Programming Language: Python
  
  Data Processing: Pandas, NumPy
  
  Machine Learning: Scikit-learn
  
  Environment Management: Anaconda, pip

Data Preprocessing
  Date Handling: Parse dates and extract features related to sales: weekends, paydays, starts & ends of year/season/month, etc.
  
  Missing Value Imputation: Forward and backward filling for oil prices
  
  Data Integration: Merge all external data sources
  
  Outlier Handling: Identify and remove data of particular outlier stores

Feature Engineering
Temporal Features: Year, month, day, week, quarter, weekend flag

Lag Features: Lagged sales values (1 day ago, 7 days ago, etc.)

Rolling Statistics: Moving averages, moving standard deviations

Holiday Features: transferred holidays, create holiday features by type(national, regional, etc.)

Store Features: Store type, city, state, cluster encoding

Modeling
Baseline Model: Random Forest Regressor

Model Ensemble: Weighted averaging of multiple models

