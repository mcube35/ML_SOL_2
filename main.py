# 관련 링크: https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting

# ============================================
# 임포트 & 전역변수
# ============================================
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import plot

import optimizer
from path_helper import PROJECT_DIR

USE_GPU = True
RANDOM_SEED = 100
OPTIMIZER_TIME_OUT = 1200
CATBOOST_MODEL_PTH = PROJECT_DIR / "model" / "catboost_model.cbm"
XGBOOST_MODEL_PTH = PROJECT_DIR / "model" / "xgb_model.json"
LIGHTGBM_MODEL_PTH = PROJECT_DIR / "model" / "lightgbm_model.txt"
# ============================================
# 1. 데이터 전처리
# ============================================

# -----------
# 데이터 로드 및 컬럼 처리
# -----------
df = pd.read_csv(PROJECT_DIR / 'dataset' / 'sales_data.csv')


# -----------
# 결측치 탐지
# -----------
print("========= 결측치 체크 =========")
print(df.isna().sum().sort_values(ascending=False))

# -----------
# Date 파생 변수 생성
# -----------
df["Date"] = pd.to_datetime(df["Date"])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek

df.drop(labels=["Date"], axis=1, inplace=True)

# -----------
# 순서형 인코딩
# -----------
order = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
df["Seasonality"] = df["Seasonality"].map(order)

# -----------
# 라벨 인코딩 (카테고리)
# -----------
cat_cols = ["Region", "Category", "Weather Condition", "Product ID", "Store ID"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    df[col] = df[col].astype("category")

# -----------
# 수치형 데이터들 표준화
# -----------
scar_cols = ['Price', 'Units Ordered', 'Units Sold', 'Inventory Level', 'Competitor Pricing']
scaler = StandardScaler()
df[scar_cols] = scaler.fit_transform(df[scar_cols])

print(df.head())

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
# XGBoost 실행
# -------------------
print("========= XGBoost 실행 =========")

import xgboost as xgb
xgb_model = None

if os.path.exists(XGBOOST_MODEL_PTH):
    print(f"📂 기존 모델 발견: {XGBOOST_MODEL_PTH}")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGBOOST_MODEL_PTH)
    print("✅ 모델 로드 완료")
else:
    print("🚀 새 모델 학습 시작")

    print("🔧 하이퍼파라미터 튜닝을 시작합니다...")
    study = optimizer.optimize_xgb(X_train=X_train, use_cuda=USE_GPU, seed=RANDOM_SEED, y_train=y_train, timeout=OPTIMIZER_TIME_OUT)

    print("🔧 최적화된 하이퍼파라미터로 학습합니다...")
    params = study.best_params.copy()
    params.update(
        n_estimators=300,
        enable_categorical=True,
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
    )
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(X_train, y_train)
    print("🔧 학습완료! 모델파일을 저장중입니다...")

    os.makedirs(os.path.dirname(XGBOOST_MODEL_PTH), exist_ok=True)
    xgb_model.save_model(XGBOOST_MODEL_PTH)
    print("💾 모델 저장 완료")

dtest = xgb.DMatrix(X_test, enable_categorical=True)
xgb_y_pred = xgb_model.get_booster().predict(dtest, validate_features=False)


# -------------------
# LightGBM 실행
# -------------------
print("========= LightGBM 실행 =========")

import lightgbm
lgbm_model = None

if os.path.exists(LIGHTGBM_MODEL_PTH):
    print(f"📂 기존 모델 발견: {LIGHTGBM_MODEL_PTH}")
    lgbm_model = lightgbm.Booster(model_file=LIGHTGBM_MODEL_PTH)
    print("✅ 모델 로드 완료")
else:
    print("🚀 새 모델 학습 시작")

    print("🔧 하이퍼파라미터 튜닝을 시작합니다...")
    study = optimizer.optimize_lgbm(X_train=X_train, y_train=y_train, use_gpu=USE_GPU, categorical_feature=cat_cols, timeout=OPTIMIZER_TIME_OUT)
    params = study.best_params.copy()
    params.update({
        "verbose": 100,
        "random_state": RANDOM_SEED,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "device": "gpu" if USE_GPU else "cpu",
        "force_row_wise": True,
        "max_depth": -1,
    })
    params.update(n_estimators=300)

    print("🔧 최적화된 하이퍼파라미터로 학습합니다...")
    lgbm_model = lightgbm.LGBMRegressor(**params)
    lgbm_model.fit(X_train, y_train, categorical_feature=cat_cols)

    print("🔧 학습완료! 모델파일을 저장중입니다...")
    os.makedirs(os.path.dirname(LIGHTGBM_MODEL_PTH), exist_ok=True)
    lgbm_model = lgbm_model.booster_
    lgbm_model.save_model(LIGHTGBM_MODEL_PTH)
    print("💾 모델 저장 완료")

lgbm_y_pred = lgbm_model.predict(X_test)

# -------------------
# CatBoost 실행
# -------------------
print("========= CatBoost 실행 =========")
import catboost 

cat_model = None

if os.path.exists(CATBOOST_MODEL_PTH):
    print(f"📂 기존 모델 발견: {CATBOOST_MODEL_PTH}")
    cat_model = catboost.CatBoostRegressor()
    cat_model.load_model(CATBOOST_MODEL_PTH)
    print("✅ 모델 로드 완료")
else:
    print("🚀 새 모델 학습 시작")

    print("🔧 하이퍼파라미터 튜닝을 시작합니다...")
    study = optimizer.optimize_catboost(
        X_train=X_train, y_train=y_train,
        cat_features=cat_cols,
        timeout=OPTIMIZER_TIME_OUT,
        use_gpu=USE_GPU,
        seed=RANDOM_SEED
    )
    params = study.best_params.copy()
    params.update({
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        "verbose": 100,
    })
    if USE_GPU:
        params.update({"task_type": "GPU", "devices": "0"})
    params.update(iterations=300)

    print("🔧 최적화된 하이퍼파라미터로 학습합니다...")
    cat_model = catboost.CatBoostRegressor(**params)
    cat_model.fit(X_train, y_train, cat_features=cat_cols)

    print("🔧 학습완료! 모델파일을 저장중입니다...")
    os.makedirs(os.path.dirname(CATBOOST_MODEL_PTH), exist_ok=True)
    cat_model.save_model(CATBOOST_MODEL_PTH)
    print("💾 모델 저장 완료")

cat_y_pred = cat_model.predict(X_test)

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
        'MAPE': f'{mape * 100:.2f}%'
    }

pred_list = [("CatBoost", cat_y_pred), ("XGBooster", xgb_y_pred), ("LightGBM", lgbm_y_pred)]
results = []
for name, pred in pred_list:
    results.append(conv_pred2dict(pred, y_test, name))

df_result = pd.DataFrame(results)
plot.performance_table(df_result)

# ----------------------------------------------
# 각 모델의 주요변수 출력하기
# ----------------------------------------------
cat_importances = cat_model.get_feature_importance(data=catboost.Pool(X_train, label=y_train, cat_features=cat_cols))
importances = [
    ("CatBoost", cat_importances),
    ("XGBoost", xgb_model.feature_importances_),
    ("LightGBM", lgbm_model.feature_importance()),
]
plot.feature_importance(importances, X_train.columns)