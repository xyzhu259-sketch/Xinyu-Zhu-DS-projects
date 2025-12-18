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
  ├── data/                          # data file
  │   ├── train.csv                  # 训练数据
  │   ├── test.csv                   # 测试数据
  │   ├── oil.csv                    # 油价数据
  │   ├── stores.csv                 # 商店信息
  │   └── holidays_events.csv        # 节假日数据
  │
  ├── src/                           # 源代码目录
  │   ├── preprocessing.py           # 数据预处理
  │   ├── feature_engineering.py     # 特征工程
  │   ├── modeling.py                # 模型训练
  │   └── utils.py                   # 工具函数
  │
  ├── models/                        # 保存的模型
  │   └── best_model.pkl
  │
  ├── submissions/                   # 提交文件
  │   └── submission.csv
  │
  ├── notebooks/                     # Jupyter笔记本
  │   ├── EDA.ipynb                  # 探索性数据分析
  │   └── Modeling.ipynb             # 建模实验
  │
  ├── requirements.txt               # 依赖包列表
  ├── README.md                      # 项目说明
  └── main.py                        # 主程序入口

