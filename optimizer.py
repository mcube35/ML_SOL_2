import numpy as np
import optuna
from typing import Sequence, Optional
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ---------------------------
# XGBoost
# ---------------------------
import xgboost as xgb

def optimize_xgb(
    X_train, y_train,
    n_trials: int = 30,
    timeout: int | None = None,
    use_cuda: bool = False,
    seed: int = 100,
) -> optuna.Study:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "device": "cuda" if use_cuda else "cpu",
            "random_state": seed,
            "tree_method": "hist",
            "n_estimators": 300,

            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 3),
            "subsample": trial.suggest_float("subsample", 0.8, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 0.95),
        }
        params.update(enable_categorical=True)

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        dval = xgb.DMatrix(X_val, enable_categorical=True)
        y_pred = model.get_booster().predict(dval, validate_features=False)
        return rmse(y_val, y_pred)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name="xgb-opt",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study

# ---------------------------
# LightGBM
# ---------------------------
import lightgbm as lgbm
from typing import Optional

def optimize_lgbm(
    X_train, y_train,
    n_trials: int = 30,
    categorical_feature = "auto",
    timeout: Optional[int] = None,
    use_gpu: bool = False,
    seed: int = 100,
) -> optuna.Study:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "boosting_type": "gbdt",
            "random_state": seed,
            "n_jobs": -1,
            "device": "gpu" if use_gpu else "cpu",
            "force_row_wise": True,
            "max_depth": -1,

            "num_leaves": trial.suggest_int("num_leaves", 55, 70),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 2),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 40, 55),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.5, 1.5),
        }
        params.update(n_estimators=300)

        model = lgbm.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, categorical_feature=categorical_feature)

        y_pred = model.predict(X_val)
        return rmse(y_val, y_pred)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name="lgbm-opt",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study

# ---------------------------
# CatBoost
# ---------------------------
from typing import Optional, Sequence
from catboost import CatBoostRegressor, Pool

def optimize_catboost(
    X_train, y_train,
    cat_features: Optional[Sequence[int]] = None,
    n_trials: int = 30,
    timeout: int | None = None,
    use_gpu: bool = False,
    seed: int = 100,
) -> optuna.Study:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            # 1) 트리 + lr
            "depth": trial.suggest_int("depth", 6, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),

            # 2) 정규화 & 손실
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 4.0, 6.0),
            "random_seed": seed,
            "allow_writing_files": False,

            # 4) 서브샘플링
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),

            # 6) 연속형 bin 수 축소
            "border_count": trial.suggest_categorical("border_count", [64, 128]),
        }
        if use_gpu:
            params.update(task_type="GPU", devices="0")

        params.update(iterations=300)

        model = CatBoostRegressor(**params)

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        valid_pool = Pool(X_val, y_val, cat_features=cat_features)

        model.fit(train_pool, eval_set=valid_pool, verbose=False)

        return model.best_score_['validation']['RMSE']

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name="catboost-opt",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study