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
# будуємо класичну лінійну регресію (OLS): модель мінімізує суму квадратів похибок (MSE).
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
# Ridge (L2): додає до втрат суму квадратів коефіцієнтів: α * ||w||_2^2
# Lasso (L1): додає до втрат суму модулів коефіцієнтів: α * ||w||_1
# Обидві зменшують "розмах" коефіцієнтів і борються з мультиколінеарністю/overfitting.
# Ridge зменшує всі коефіцієнти плавно; Lasso може занулювати деякі (фічер-селекція).
# Для підбору сили штрафу α використаємо крос-валідацію (CV) на сітці.
# ---------------------------------------------------------------

print('ЕТАП 8: НАВЧАННЯ RidgeCV (L2) та LassoCV (L1) з підбором alpha')

# сітка можливих α (логарифмічна шкала)
alphas = np.logspace(-3, 3, 21)  # від 0.001 до 1000

# RidgeCV: за замовчуванням оптимізує за MSE через cross-val
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train_scaled, y_train)

# LassoCV: підбирає α, тут бажано збільшити max_iter на випадок повільної збіжності
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

print('ЕТАП 12: ВІЗУАЛІЗАЦІЯ ПОРІВНЯННЯ МЕТРИК')
metrics_table = pd.DataFrame({
    "Model": ["OLS", "Ridge", "Lasso"],
    "MSE":   [mse, mse_ridge, mse_lasso],
    "MAE":   [mae, mae_ridge, mae_lasso],
    "R2":    [r2,  r2_ridge,  r2_lasso],
})
print('\nЗведена таблиця метрик:\n', metrics_table)
