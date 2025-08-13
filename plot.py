import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title(f"{model_name} - Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def pred_vs_actual(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def performance_table(df_result: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(df_result)))
    ax.axis('off')  # 축 제거

    # 테이블 추가
    table = ax.table(
        cellText=df_result.round(4).values,
        colLabels=df_result.columns,
        cellLoc='center',
        loc='center'
    )

    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    plt.title("Models Performance", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def feature_importance(importances: list, X_columns: list):
    for name, importance in importances:
        df_feat = pd.DataFrame({
            'Feature': X_columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(8, max(4, len(df_feat) * 0.3)))
        sns.barplot(x="Importance", y="Feature", data=df_feat, hue=X_columns, legend=False, palette="viridis")
        plt.title(f"{name} - Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.show()