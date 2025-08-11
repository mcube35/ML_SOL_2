# 관련 링크: https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting

# ============================================
# 임포트 & 전역변수
# ============================================
import pandas as pd
import numpy as np
import plot
from path_helper import PROJECT_DIR

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
RANDOM_SEED = 100

# ============================================
# 1. 데이터 전처리
# ============================================

# -----------
# 데이터 로드 및 불필요한 컬럼 제거
# -----------
df = pd.read_csv(PROJECT_DIR / 'dataset' / 'sales_data.csv')
df.drop(labels=["Date"], axis=1, inplace=True)

# -----------
# 결측치 탐지
# -----------
print(df.isna().sum().sort_values(ascending=False))
print("========= 결측치 체크 =========")

# -----------
# 순서형 라벨 인코딩
# -----------
order = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
df["Seasonality"] = df["Seasonality"].map(order)

# -----------
# 라벨 인코딩 (카테고리)
# -----------
cat_cols = ["Product ID", "Store ID", "Region", "Category", "Weather Condition"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    df[col] = df[col].astype("category")



# ============================================
# 2. 모델 학습시키고 test셋을 이용해서 예측결과(y_pred) 뽑기
# ============================================

# -------------------
# Train/Test셋 만들기
# -------------------
X = df.drop(columns='Demand')
y = df["Demand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# -------------------
# CatBoost 학습
# -------------------
import catboost
cat_model = catboost.CatBoostRegressor(
    iterations=2500,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5.0,
    bagging_temperature=0.5,
    border_count=128,

    verbose=100,
    cat_features=cat_cols,
    random_state=RANDOM_SEED
)
cat_model.fit(X_train, y_train)
cat_y_pred = cat_model.predict(X_test)

# -------------------
# XGBoost 학습
# -------------------
import xgboost

xgb_model = xgboost.XGBRegressor(
    n_estimators=2500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    device = "cuda",
    tree_method = "hist",

    enable_categorical=True,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

# -------------------
# LightGBM 학습
# -------------------
import lightgbm

lgbm_model = lightgbm.LGBMRegressor(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    n_estimators=2500,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,

    random_state=RANDOM_SEED
)
lgbm_model.fit(X_train, y_train)
lgbm_y_pred = lgbm_model.predict(X_test)



# ============================================
# 3. 시각화 하기
# ============================================

# -----------------------------------------------
# 각 모델의 MAE, MSE, RMSE, MAPE 도출하고 시각화하기
# ----------------------------------------------
def conv_pred2dict(y_pred, y_test, name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

pred_list = [(cat_y_pred, "CatBoost"), (xgb_y_pred, "XGBooster"), (lgbm_y_pred, "LightGBM")]
results = []
for pred, name in pred_list:
    results.append(conv_pred2dict(pred, y_test, name))

df_result = pd.DataFrame(results)
plot.performance_table(df_result)

# ----------------------------------------------
# 각 모델의 주요변수 출력하기
# ----------------------------------------------
models = [
    ("CatBoost", cat_model),
    ("XGBoost", xgb_model),
    ("LightGBM", lgbm_model),
]
plot.feature_importance(models, X_train.columns)