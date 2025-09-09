import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# завантаження даних із CSV
data = pd.read_csv("WineQT.csv")

# перевіримо, а що всередині
print('ЕТАП 0: ЗАВАНТАЖИЛИ ДАТАСЕТ')
print(data.info(True))

# Видалимо непотрібну колонку Id
data = data.drop("Id", axis='columns')

print('ЕТАП 1: ОЧИСТИЛИ ДАНІ ВІД КОЛОНКИ ID ОСТАННЬОЇ')
print(data.info(True))

print('ЕТАП 2: ПОШУК КОРЕЛЯЦІЙ ЗА ДОПОМОГОЮ КОРЕЛЯЦІЙНОЇ МАТРИЦІ')

# побудова кореляційної матриці
plt.figure(figsize=(14,10))

# тут виклик data.corr() треба для того щоб витягти з data frame pairwise correlation (попарну кореляцію)
# стандартно використовуються коефіцієнти Пірсона
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Кореляція між різними ознаками датасету")
plt.show()

# В датасеті нема price тому будемо пробувати робити лінійну регресію оцінку якості
print('ЕТАП 3: ГОТУЄМО ВХІДНІ ЗМІННІ (ЕСТИМАТОРИ) ТА ЦІЛЬ У ЯКОСТІ оцінки Quantity')

# на вхід надходять всі колонки окрім якості яку треба оцінити
X = data.drop("quality", axis='columns')
y = data["quality"]

print('ЕТАП 4: ФОРМУЄМО ТРЕНУВАЛЬНІ ТА НАВЧАЛЬНІ ДАТАСЕТИ ТА МАСШТАБУЄМО')
# ділимо обидві множини на тренувальні та для тестування (використаємо 80/20 пропорцію)
random_shuffle = 47
size_for_testing = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_for_testing, random_state=random_shuffle)

# на вході маємо ознаки в різних масштабах (alcohol~10, а sulphates~0.5)
# щоб коефіцієнти були сумісні між собою потрібно масштабувати
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('ЕТАП 5: НАВЧАННЯ МОДЕЛІ')
# створюємо та навчаємо модель
# будуємо класичну лінійну регресію (OLS ordinary least squares): модель мінімізує суму квадратів похибок (MSE).
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print('ЕТАП 6: ВИКОРИСТАННЯ МОДЕЛІ')
# прогноз на тестовому датасеті
y_pred = model.predict(X_test_scaled)

print('ЕТАП 7: АНАЛІЗ РЕЗУЛЬТАТІВ')

# Метрики якості
# MSE (середньоквадратична похибка) – «штрафна функція» моделі.
# MAE (середня абсолютна похибка) – «середній відхил у балах».
# R² (коефіцієнт детермінації) – наскільки добре модель пояснює варіацію у даних (0 → погано, 1 → ідеально).

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# вплив ознак (коефіцієнти моделі)
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print('\nКоефіцієнти OLS:\n', coefficients)

# ---------------------------------------------------------------
# ЕТАП 8: ДОДАЄМО ДВІ ШТРАФНІ ФУНКЦІЇ (РЕГУЛЯРИЗАЦІЮ)
# Ridge (L2): додає до втрат суму квадратів коефіцієнтів: alpha * ||w||_2^2
# Lasso (L1): додає до втрат суму модулів коефіцієнтів: alpha * ||w||_1
# Обидві зменшують "розмах" коефіцієнтів і борються з мультиколінеарністю/overfitting.
# Ridge зменшує всі коефіцієнти плавно; Lasso може занулювати деякі (фічер-селекція).
# Для підбору сили штрафу alpha використаємо крос-валідацію (CV) на сітці.
# ---------------------------------------------------------------

print('ЕТАП 8: НАВЧАННЯ RidgeCV (L2) та LassoCV (L1) з підбором alpha')

# сітка можливих alpha (логарифмічна шкала)
alphas = np.logspace(-3, 3, 21)  # від 0.001 до 1000

# RidgeCV: за замовчуванням оптимізує за MSE через cross-val
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train_scaled, y_train)

# LassoCV: підбирає alpha, тут бажано збільшити max_iter на випадок повільної збіжності
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=random_shuffle)
lasso.fit(X_train_scaled, y_train)

print(f"Найкраще alpha для Ridge: {ridge.alpha_:.5f}")
print(f"Найкраще alpha для Lasso: {lasso.alpha_:.5f}")

print('ЕТАП 9: ПРОГНОЗИ ТА МЕТРИКИ ДЛЯ RIDGE ТА LASSO')
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print('ЕТАП 10: ПОРІВНЯННЯ МЕТРИК (менше — краще для MSE/MAE; більше — краще для R²)')
print(f"Ridge | alpha={ridge.alpha_:.5f} -> MSE: {mse_ridge:.3f}, MAE: {mae_ridge:.3f}, R²: {r2_ridge:.3f}")
print(f"Lasso | alpha={lasso.alpha_:.5f} -> MSE: {mse_lasso:.3f}, MAE: {mae_lasso:.3f}, R²: {r2_lasso:.3f}")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "OLS": model.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_,
}).set_index("Feature").sort_index()

print('ЕТАП 11: Порівняння коефіцієнтів (OLS vs Ridge vs Lasso):\n', coef_df)

# Скільки коефіцієнтів занулив Lasso (фічер-селекція)?
n_zeros_lasso = np.sum(np.isclose(lasso.coef_, 0.0))
print(f"Lasso занулило коефіцієнтів: {n_zeros_lasso} із {len(lasso.coef_)}")

print('ЕТАП 12: ВІЗУАЛІЗАЦІЯ ТА ПОРІВНЯННЯ МЕТРИК')
metrics_table = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso"],
    "MSE": [mse, mse_ridge, mse_lasso],
    "MAE": [mae, mae_ridge, mae_lasso],
    "R2": [r2, r2_ridge, r2_lasso],
})
print('\nЗведена таблиця метрик:\n', metrics_table)

print('ЕТАП 13: ПАРИТЕТНІ ГРАФІКИ (ŷ проти y) ДЛЯ КОЖНОЇ МОДЕЛІ')


def parity_plot(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=25)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle='--')  # ідеальна лінія y=x
    plt.xlabel("Факт (y)")
    plt.ylabel("Передбачення (ŷ)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


parity_plot(y_test, y_pred, "Parity plot — OLS (ŷ vs y)")
parity_plot(y_test, y_pred_ridge, f"Parity plot — Ridge (α={ridge.alpha_:.3g})")
parity_plot(y_test, y_pred_lasso, f"Parity plot — Lasso (α={lasso.alpha_:.3g})")

print('ЕТАП 14: ДЕМОНСТРАЦІЯ, ДЕ МОДЕЛЬ ПОМИЛЯЄТЬСЯ')


def residual_plots(y_true, y_pred, title_prefix):
    resid = y_true - y_pred

    # 14.1 Гістограма похибок
    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=20)
    plt.axvline(0, linestyle='--')
    plt.title(f"{title_prefix}: розподіл похибок (y - ŷ)")
    plt.xlabel("Похибка")
    plt.ylabel("К-сть")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 14.2 Похибка vs Передбачення
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, alpha=0.5, s=20)
    plt.axhline(0, linestyle='--')
    plt.title(f"{title_prefix}: похибка vs передбачення")
    plt.xlabel("Передбачення (ŷ)")
    plt.ylabel("Похибка (y - ŷ)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


residual_plots(y_test, y_pred, "OLS")
residual_plots(y_test, y_pred_ridge, f"Ridge (α={ridge.alpha_:.3g})")
residual_plots(y_test, y_pred_lasso, f"Lasso (α={lasso.alpha_:.3g})")

print("ЕТАП 15: Лінії регресії по окремих фічах (OLS/Ridge/Lasso)")

# які фічі показувати
features_to_plot = ["alcohol", "sulphates", "volatile acidity", "density"]

# середні значення по train (щоб «заморозити» інші ознаки)
X_train_means = X_train.mean()


def predict_line_for_feature(feature_name, model_obj, n_points=200):
    x_min = X[feature_name].min()
    x_max = X[feature_name].max()
    grid = np.linspace(x_min, x_max, n_points)

    # Базова матриця ознак із середніми значеннями
    X_line = pd.DataFrame([X_train_means.values] * n_points, columns=X.columns)
    X_line[feature_name] = grid

    # Масштабуємо і прогнозуємо
    X_line_scaled = scaler.transform(X_line)
    y_line = model_obj.predict(X_line_scaled)
    return grid, y_line


for feat in features_to_plot:
    # Фактичні точки з тесту по обраній фічі
    plt.figure(figsize=(7, 5))
    plt.scatter(X_test[feat], y_test, alpha=0.55, s=25, label="Факт (test)")

    # Лінії трьох моделей
    gx, gy_ols = predict_line_for_feature(feat, model)
    _, gy_ridge = predict_line_for_feature(feat, ridge)
    _, gy_lasso = predict_line_for_feature(feat, lasso)

    order = np.argsort(gx)
    plt.plot(gx[order], gy_ols[order], '-', label='OLS')
    plt.plot(gx[order], gy_ridge[order], '--', label=f'Ridge (α={ridge.alpha_:.3g})')
    plt.plot(gx[order], gy_lasso[order], ':', label=f'Lasso (α={lasso.alpha_:.3g})')

    plt.title(f"Якість vs {feat}: дані (test) та лінії регресії")
    plt.xlabel(feat)
    plt.ylabel("quality")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
