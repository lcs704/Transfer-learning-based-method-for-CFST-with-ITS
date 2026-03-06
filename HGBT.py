# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"shape: {df.shape}")
    print("\ndata:")
    print(df.head())
    return df


def preprocess_data(df):
    feature_cols = ['D', 't', 'L', 'r', 'fcu', 'fy', 'e', 'cross-section type']
    X = df[feature_cols]
    y = df['N']
    domain = df['is_CFST'].values

    Xs, ys = X[domain == 0], y[domain == 0]
    Xt, yt = X[domain == 1], y[domain == 1]
    print(f"source dataset: {len(ys)}")
    print(f"target dataset: {len(yt)}")

    return Xs, ys, Xt, yt

def plot_predictions(y_train, y_train_pred, y_test, y_pred):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    y_all = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_pred])
    residuals = y_all - y_all_pred
    indices = np.arange(1, len(y_all)+1)
    train_len = len(y_train)

    # training set
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))

    # testing set
    r2_test = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((y_test - y_pred) / y_test))

    # total set
    r2_total = r2_score(y_all, y_all_pred)
    mae_total = mean_absolute_error(y_all, y_all_pred)
    mse_total = mean_squared_error(y_all, y_all_pred)
    rmse_total = np.sqrt(mse_total)
    mape_total = np.mean(np.abs((y_all - y_all_pred) / y_all))

    results = pd.DataFrame({
        "r2": [r2_train, r2_test, r2_total],
        "rmse": [rmse_train, rmse_test, rmse_total],
        "mae": [mae_train, mae_test, mae_total],
        "mape": [mape_train, mape_test, mape_total]
    }, index=["Training", "Testing", "Total"])

    print(results.to_string(float_format="%.3f"))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axvspan(0, train_len, facecolor='mistyrose', alpha=1, zorder=0.5)  # 训练集
    ax.axvspan(train_len, len(y_all)+1, facecolor='honeydew', alpha=1, zorder=0.5)  # 测试集

    ax.plot(indices, y_all, color='black', linestyle='-', marker='', label='Experimental Value')
    ax.plot(indices[:train_len], y_all_pred[:train_len], color='red', label='Training Set Predicted Value')
    ax.plot(indices[train_len-1:], y_all_pred[train_len-1:], color='blue', label='Testing Set Predicted Value')

    ax.bar(indices[:train_len], residuals[:train_len], color='red', alpha=0.5, width=0.8, bottom=0, label='Training Set Residuals')
    ax.bar(indices[train_len:], residuals[train_len:], color='blue', alpha=0.5, width=0.8, bottom=0, label='Testing Set Residuals')


    ax.axhline(0, color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Sample number')
    ax.set_ylabel('$N_u$ (kN)')
    ax.legend(loc=2,
              ncol=2,
              fontsize=20)
    ax.grid(False)

    ax.set_xlim(0, 42)
    ax.set_xticks([1, 10, 20, 30, 41])
    ax.set_yticks(np.arange(-1000, 5001, 1000))

    ax_right = ax.twinx()
    ax_right.set_ylabel('Residuals (kN)')
    ax_right.set_ylim(ax.get_ylim())
    ax_right.yaxis.set_major_locator(ax.yaxis.get_major_locator())
    ax_right.grid(False)

    ax.tick_params(axis='both', direction='in', length=6, width=2, colors='black')
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax_right.tick_params(axis='both', direction='in', length=6, width=2)

    train_center = (0 + train_len - 1) / 2
    test_center = (train_len + len(y_all) - 1) / 2
    ax.text(train_center, -500, f'Training set R² = {r2_train:.3f}',
            ha='center', va='top')
    ax.text(test_center, -500, f'Testing set R² = {r2_test:.3f}',
            ha='center', va='top')

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, alpha=0.8, color='blue', s=120, marker='o', edgecolors='black', linewidths=0.5, label='Training Set')
    plt.scatter(y_test, y_pred, alpha=0.8, color='red', s=120, marker='^', edgecolors='black', linewidths=0.5, label='Testing Set')

    max_val = 3000
    plt.plot([0, max_val + 500], [0, max_val + 500], 'k-', lw=2, zorder=0.5)
    x_points = np.array([0, max_val + 500])
    plt.xlim(0, max_val + 500)
    plt.ylim(0, max_val + 500)

    y1_points = 1.2 * x_points
    y2_points = 0.8 * x_points
    plt.plot(x_points, y1_points, 'k--', lw=1.5, alpha=0.7, label='±20% error', zorder=0.5)
    plt.plot(x_points, y2_points, 'k--', lw=1.5, alpha=0.7, zorder=0.5)

    y3_points = 1.4 * x_points
    y4_points = 0.6 * x_points
    plt.plot(x_points, y3_points, 'k-.', lw=1.5, alpha=0.7, label='±40% error', zorder=0.5)
    plt.plot(x_points, y4_points, 'k-.', lw=1.5, alpha=0.7, zorder=0.5)

    plt.xlabel('Experimental value of $N_u$ (kN)')
    plt.ylabel('Predicted value of $N_u$ (kN)')
    plt.legend(loc=4,
              ncol=1,
              fontsize=22)
    plt.grid(False)

    plt.tick_params(axis='both', direction='in', length=6, width=2)
    plt.tick_params(bottom=True, top=False, left=True, right=False)

    metrics_text = (
        f"Total Set\n"
        f"R² = {r2_total:.3f}\n"
        f"RMSE = {rmse_total:.3f}\n"
        f"MAE = {mae_total:.3f}\n"
        f"MAPE = {mape_total*100:.3f} %"
    )

    plt.text(
        0.05, 0.95, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=22,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.show()

def main():
    file_path = "The source dataset and target dataset.xlsx"
    sheet_name = "Sheet1"
    df = load_data(file_path, sheet_name)
    Xs, ys, Xt, yt = preprocess_data(df)

    best_params = {
        "max_iter": 217,
        "learning_rate": 0.6357,
        "max_leaf_nodes": 255,
        "min_samples_leaf": 1,
        "l2_regularization": 1.2534,
        "max_bins": 130,
        "random_state": 42
    }

    X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.5, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = HistGradientBoostingRegressor(**best_params)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    plot_predictions(y_train, y_train_pred, y_test, y_test_pred)

    print("\n=== finish ===")


if __name__ == "__main__":
    main()