Store Sales - Time Series Forecasting 
  -- Data analysis and prediction using Random Forest Regression by Xinyu Zhu

Project Description:
    The Store Sales competition is a competitive project from Kaggle, aiming to predict the Ecuadorian store Favorita's sales using given past data regarding goods information, time and oil price. 
    The project involves data prediction with time & date related data, feature engineering and the use of machine learning algorithms.\

Competition information:
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
  ├── README.md                      # Project description
  ├── .gitignore
  └── LICENSE                    # MIT license

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

Rolling Statistics: Sales averages, sales standard deviations

Holiday Features: transferred holidays, create holiday features by type(national, regional, etc.)

Store Features: Store type, city, state 

Store bands: classification of stores into groups based on historical mean sales

Modeling
  Baseline Model: Random Forest Regressor
  
  Model Ensemble: Using of multiple models on classified data groups based on store bands, using weighted average to conclude final prediction

Key Features discovered
  1. Onpromotion --whether the goods are sold at lower prices
  2. dcoilwtico --price of oil on the particular date
  3. Store number --the specific store where the goods are sold
  4. Holiday events --Whether holidays are taking place in the region on the day predicted
  5. Family --The type of good being sold

Final Performance of prediction
  Score of predicted data: 0.49473
  Place on the leaderboard: 175/667 (Recorded on Dec 17 2025)

References
  1. Kaggle competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
  2. Store Sales - Linear regression & Random Forest, R.Coletto, 3 years ago, https://www.kaggle.com/code/rcoletto/store-sales-linear-regression-random-forest

License
  This project is licensed under the MIT License.

Acknowledgments
  Kaggle platform for hosting the competition and providing data
  Kaggle discussion posts for sharing example approaches to the Store Sales competition

Contact
  Author: Xinyu Zhu
  Email: xyzhu259@uw.edu
  Kaggle: https://www.kaggle.com/xinyuzhu259
