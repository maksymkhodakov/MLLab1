import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# ЕТАП 0: ЗАВАНТАЖЕННЯ ТА ПЕРЕВІРКА ДАНИХ
# -------------------------------------------------------------------
# CSV має колонки (приклад):
# fixed acidity, volatile acidity, ..., sulphates, alcohol, quality, Id
data = pd.read_csv("WineQT.csv")
print("ЕТАП 0: ЗАВАНТАЖИЛИ ДАТАСЕТ")
print(data.info(True))  # дивимось типи даних, наявність пропусків тощо

# Видаляємо службовий стовпчик Id (він просто індекс/ідентифікатор, не ознака)
if "Id" in data.columns:
    data = data.drop("Id", axis="columns")

print("\nЕТАП 1: ОЧИСТИЛИ ДАНІ (видалили Id за наявності)")
print(data.info(True))

# -------------------------------------------------------------------
# ЕТАП 2: КОРЕЛЯЦІЙНА МАТРИЦЯ (розвідковий аналіз)
# -------------------------------------------------------------------
# Корисно побачити, як корелює ціль 'alcohol' із іншими ознаками.
print("\nЕТАП 2: ПОШУК КОРЕЛЯЦІЙ (heatmap)")
plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Кореляція між ознаками (включно з alcohol)")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# ЕТАП 3: ФОРМУЄМО ОЗНАКИ (X) ТА ЦІЛЬ (y)
# -------------------------------------------------------------------
# Ціль (неперервна): alcohol
# У X НЕ має бути alcohol (бо ми її прогнозуємо)
print("\nЕТАП 3: ГОТУЄМО ВХІДНІ ЗМІННІ (X) ТА ЦІЛЬ (y = alcohol)")
y = data["alcohol"].astype(float)
X = data.drop(columns=["alcohol"])

# -------------------------------------------------------------------
# ЕТАП 4: РОЗПОДІЛ НА TRAIN / VAL / TEST + МАСШТАБУВАННЯ
# -------------------------------------------------------------------
print("\nЕТАП 4: РОЗПОДІЛ НА TRAIN / VAL / TEST ТА МАСШТАБУВАННЯ")

RANDOM_STATE = 47
train_size = 0.6
val_size = 0.2
test_size = 0.2
assert abs(train_size + val_size + test_size - 1.0) < 1e-9

# 4.1) Спочатку відокремлюємо TRAIN від тимчасового набору (VAL+TEST)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(1 - train_size), random_state=RANDOM_STATE
)

# 4.2) Ділимо тимчасовий набір на VAL і TEST так, щоб сумарно було 20/20
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=test_size / (test_size + val_size),
    random_state=RANDOM_STATE
)

# 4.3) Стандартизація (fit тільки на TRAIN, transform на VAL/TEST)
# z = (x - μ_train) / σ_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # обчислюємо μ,σ на train і масштабуємо train
X_val_scaled = scaler.transform(X_val)  # масштабуємо val тими ж μ,σ
X_test_scaled = scaler.transform(X_test)  # масштабуємо test тими ж μ,σ

print(f"Форми: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")


# -------------------------------------------------------------------
# ДОПОМІЖНІ ФУНКЦІЇ: метрики + графіки + звіт про overfitting
# -------------------------------------------------------------------
def print_reg_metrics(name, y_true, y_pred):
    """
    Обчислює й друкує класичні метрики регресії:
    - MSE = (1/m) Σ (y_i - ŷ_i)^2
    - MAE = (1/m) Σ |y_i - ŷ_i|
    - R²  = 1 - SS_res/SS_tot

    Повертає словник для подальшого аналізу.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>10} | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return {"MSE": mse, "MAE": mae, "R2": r2}


def parity_plot(y_true, y_pred, title):
    """
    Parity plot (ŷ vs y): наскільки передбачення співпадають із фактами.
    Ідеально всі точки лежать на лінії y = x (пунктир).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=25)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel("Факт (alcohol)")
    plt.ylabel("Передбачення (alcohol)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def residual_plots(y_true, y_pred, title_prefix):
    """
    Показує розподіл похибок та залежність похибки від передбачення.
    Для адекватної лінійної моделі резидуали мають бути «шумом» навколо 0.
    """
    resid = y_true - y_pred

    # 1) Гістограма похибок
    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=20)
    plt.axvline(0, linestyle='--')
    plt.title(f"{title_prefix}: розподіл похибок (y - ŷ)")
    plt.xlabel("Похибка alcohol")
    plt.ylabel("К-сть")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) Похибка vs Передбачення
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, alpha=0.5, s=20)
    plt.axhline(0, linestyle='--')
    plt.title(f"{title_prefix}: похибка vs передбачення")
    plt.xlabel("Передбачення alcohol (ŷ)")
    plt.ylabel("Похибка (y - ŷ)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def overfit_report(name, m_train, m_val, r2_gap_thresh=0.05, mae_gap_rel=0.10):
    """
    Проста евристика для виявлення перенавчання:
    - Якщо R²(train) - R²(val) > r2_gap_thresh → ризик overfitting
    - Якщо MAE(val) > (1 + mae_gap_rel) * MAE(train) → ризик overfitting
    Пороги можна змінювати під ваші дані.
    """
    r2_gap = m_train["R2"] - m_val["R2"]
    mae_rel = (m_val["MAE"] - m_train["MAE"]) / max(1e-9, m_train["MAE"])

    print(f"\n[Overfitting check] {name}")
    print(f"  R² gap (train - val): {r2_gap:+.3f}  |  MAE relative gap: {mae_rel:+.1%}")

    flags = []
    if r2_gap > r2_gap_thresh:
        flags.append(f"R² gap > {r2_gap_thresh}")
    if mae_rel > mae_gap_rel:
        flags.append(f"MAE(val) на {mae_rel:.0%} більша за MAE(train)")

    if flags:
        print("  ⚠️ Ознаки overfitting → " + "; ".join(flags))
    else:
        print("  ✅ Явних ознак overfitting не виявлено")


# -------------------------------------------------------------------
# ЕТАП 5: НАВЧАННЯ OLS (без штрафу)
# -------------------------------------------------------------------
print("\nЕТАП 5: НАВЧАННЯ OLS (Ordinary Least Squares) для alcohol")
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)  # мінімізує MSE (через нормальні рівняння або QR)

# Прогнози OLS на всіх множинах
y_pred_tr_ols = ols.predict(X_train_scaled)
y_pred_va_ols = ols.predict(X_val_scaled)
y_pred_te_ols = ols.predict(X_test_scaled)

print("\nЕТАП 6: МЕТРИКИ OLS (train/val/test)")
m_ols_tr = print_reg_metrics("OLS-train", y_train, y_pred_tr_ols)
m_ols_va = print_reg_metrics(" OLS-val", y_val, y_pred_va_ols)
m_ols_te = print_reg_metrics("OLS-test", y_test, y_pred_te_ols)
overfit_report("OLS", m_ols_tr, m_ols_va)

# -------------------------------------------------------------------
# ЕТАП 7: НАВЧАННЯ RIDGE/LASSO З CV (штрафні функції L2 та L1)
# -------------------------------------------------------------------
print("\nЕТАП 7: НАВЧАННЯ RidgeCV (L2) та LassoCV (L1) для alcohol")
# Добираємо α з логарифмічної сітки (від слабого штрафу до сильного)
alphas = np.logspace(-3, 3, 21)  # 0.001 ... 1000

# RidgeCV/LassoCV всередині виконують k-fold CV на train і мінімізують MSE
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train_scaled, y_train)

lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=RANDOM_STATE)
lasso.fit(X_train_scaled, y_train)

print(f"Найкраще alpha для Ridge: {ridge.alpha_:.5f}")
print(f"Найкраще alpha для Lasso: {lasso.alpha_:.5f}")

# Прогнози Ridge/Lasso
y_pred_tr_r = ridge.predict(X_train_scaled)
y_pred_va_r = ridge.predict(X_val_scaled)
y_pred_te_r = ridge.predict(X_test_scaled)

y_pred_tr_l = lasso.predict(X_train_scaled)
y_pred_va_l = lasso.predict(X_val_scaled)
y_pred_te_l = lasso.predict(X_test_scaled)

print("\nЕТАП 8: МЕТРИКИ RIDGE (train/val/test)")
m_r_tr = print_reg_metrics("Ridge-tr", y_train, y_pred_tr_r)
m_r_va = print_reg_metrics(" Ridge-va", y_val, y_pred_va_r)
m_r_te = print_reg_metrics("Ridge-te", y_test, y_pred_te_r)
overfit_report("Ridge", m_r_tr, m_r_va)

print("\nЕТАП 9: МЕТРИКИ LASSO (train/val/test)")
m_l_tr = print_reg_metrics("Lasso-tr", y_train, y_pred_tr_l)
m_l_va = print_reg_metrics(" Lasso-va", y_val, y_pred_va_l)
m_l_te = print_reg_metrics("Lasso-te", y_test, y_pred_te_l)
overfit_report("Lasso", m_l_tr, m_l_va)

# -------------------------------------------------------------------
# ЕТАП 10: ПОРІВНЯННЯ КОЕФІЦІЄНТІВ (вплив ознак на alcohol)
# -------------------------------------------------------------------
# Інтерпретація коефіцієнтів після стандартизації:
# знак w_j показує напрям впливу (↑фіча → ↑ або ↓ alcohol),
# величина |w_j| — «сила» зв’язку в стандартних відхиленнях.
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "OLS": ols.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
}).set_index("Feature").sort_index()

print("\nЕТАП 10: Порівняння коефіцієнтів (OLS vs Ridge vs Lasso) для alcohol:")
print(coef_df)

# Скільки коефіцієнтів занулив Lasso (вбудована селекція ознак)
n_zeros_lasso = np.sum(np.isclose(lasso.coef_, 0.0))
print(f"\nLasso занулило коефіцієнтів: {n_zeros_lasso} із {len(lasso.coef_)}")

# -------------------------------------------------------------------
# ЕТАП 11: ЗВЕДЕНА ТАБЛИЦЯ МЕТРИК (для зручного зведення)
# -------------------------------------------------------------------
metrics_table = pd.DataFrame({
    "Split": ["train", "val", "test"],
    "OLS_MSE": [m_ols_tr["MSE"], m_ols_va["MSE"], m_ols_te["MSE"]],
    "OLS_R2": [m_ols_tr["R2"], m_ols_va["R2"], m_ols_te["R2"]],
    "Ridge_MSE": [m_r_tr["MSE"], m_r_va["MSE"], m_r_te["MSE"]],
    "Ridge_R2": [m_r_tr["R2"], m_r_va["R2"], m_r_te["R2"]],
    "Lasso_MSE": [m_l_tr["MSE"], m_l_va["MSE"], m_l_te["MSE"]],
    "Lasso_R2": [m_l_tr["R2"], m_l_va["R2"], m_l_te["R2"]],
})
print("\nЗведена таблиця метрик (train/val/test) для alcohol:")
print(metrics_table)

# -------------------------------------------------------------------
# ЕТАП 12: ПАРИТЕТНІ ГРАФІКИ (ŷ vs y) ДЛЯ КОЖНОЇ МОДЕЛІ — test-спліт
# -------------------------------------------------------------------
print("\nЕТАП 12: Parity (ŷ vs y) на тесті — alcohol")
parity_plot(y_test, y_pred_te_ols, "Parity — OLS (alcohol, test)")
parity_plot(y_test, y_pred_te_r, f"Parity — Ridge (α={ridge.alpha_:.3g}, alcohol, test)")
parity_plot(y_test, y_pred_te_l, f"Parity — Lasso (α={lasso.alpha_:.3g}, alcohol, test)")

# -------------------------------------------------------------------
# ЕТАП 13: RESIDUALS — де модель помиляється (test-спліт)
# -------------------------------------------------------------------
print("\nЕТАП 13: Розподіл похибок та похибка vs передбачення (test) — alcohol")
residual_plots(y_test, y_pred_te_ols, "OLS (alcohol, test)")
residual_plots(y_test, y_pred_te_r, f"Ridge (α={ridge.alpha_:.3g}, alcohol, test)")
residual_plots(y_test, y_pred_te_l, f"Lasso (α={lasso.alpha_:.3g}, alcohol, test)")

# -------------------------------------------------------------------
# ЕТАП 14: ЛІНІЇ РЕГРЕСІЇ В ПРОЄКЦІЇ ОКРЕМИХ ОЗНАК (інші на mean(train))
# -------------------------------------------------------------------
print("\nЕТАП 14: Лінії регресії для вибраних ознак (OLS/Ridge/Lasso) — таргет alcohol")
# На цих графіках змінюємо 1 ознаку в її діапазоні, інші фіксуємо на середніх (train).
# Це візуалізує «нахил» і напрямок впливу однієї ознаки, враховуючи наявність інших.
features_to_plot = ["sulphates", "volatile acidity", "residual sugar", "density"]

X_train_means = X_train.mean()  # фіксуємо інші ознаки на середніх train


def predict_line_for_feature(feature_name, model_obj, n_points=200):
    x_min = X[feature_name].min()
    x_max = X[feature_name].max()
    grid = np.linspace(x_min, x_max, n_points)

    # Базова матриця: всі ознаки = середнім train
    X_line = pd.DataFrame([X_train_means.values] * n_points, columns=X.columns)
    # Змінюємо тільки одну ознаку
    X_line[feature_name] = grid

    # Масштабуємо та прогнозуємо
    X_line_scaled = scaler.transform(X_line)
    y_line = model_obj.predict(X_line_scaled)
    return grid, y_line


for feat in features_to_plot:
    plt.figure(figsize=(7, 5))
    # Розсіяння фактів (test) у проєкції цієї ознаки
    plt.scatter(X_test[feat], y_test, alpha=0.55, s=25, label="Факт alcohol (test)")

    # Лінії трьох моделей
    gx, gy_ols = predict_line_for_feature(feat, ols)
    _, gy_ridge = predict_line_for_feature(feat, ridge)
    _, gy_lasso = predict_line_for_feature(feat, lasso)

    # Сортуємо, щоб лінії були гладкими
    order = np.argsort(gx)
    plt.plot(gx[order], gy_ols[order], '-', label='OLS')
    plt.plot(gx[order], gy_ridge[order], '--', label=f'Ridge (α={ridge.alpha_:.3g})')
    plt.plot(gx[order], gy_lasso[order], ':', label=f'Lasso (α={lasso.alpha_:.3g})')

    plt.title(f"alcohol vs {feat}: дані (test) та лінії регресії")
    plt.xlabel(feat)
    plt.ylabel("alcohol")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# ЕТАП 15: LEARNING CURVE — OLS (pipeline: scaler + OLS)
# -------------------------------------------------------------------
print("\nЕТАП 15: Learning Curve — OLS (R²) для alcohol")


def plot_learning_curve(estimator, X_all, y_all, title, cv=5, n_jobs=None):
    """
    Крива навчання показує, як змінюється якість (R²) на трені/валі при зростанні розміру train.
    Інтерпретація:
    - Великий розрив між Train і Val R² при малих train_size → можлива висока варіативність/overfit.
    - Збільшення train_size вирівнює/зближує криві → модель узагальнює краще.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_all, y_all, cv=cv, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="r2"
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, marker='o', label='Train R²')
    plt.plot(train_sizes, val_mean, marker='o', label='Val R²')
    plt.xlabel("Кількість тренувальних зразків")
    plt.ylabel("R² (вище — краще)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Пайплайн гарантує, що стандартизація робиться всередині CV коректно
pipe_ols = Pipeline([("scaler", StandardScaler()), ("mdl", LinearRegression())])
plot_learning_curve(pipe_ols, X, y, "Learning Curve — OLS (R²), target: alcohol", cv=5)

# -------------------------------------------------------------------
# ЕТАП 16: VALIDATION CURVES — Ridge/Lasso (α vs R²)
# -------------------------------------------------------------------
print("\nЕТАП 16: Validation Curves — Ridge/Lasso (α vs R²) для alcohol")
alphas_vc = np.logspace(-3, 3, 21)


def plot_validation_curve(model_cls, param_name, param_range, X_all, y_all, title, cv=5):
    """
    Показує, як змінюється якість (R²) в залежності від гіперпараметра (тут α).
    Інтерпретація:
    - Дуже малий α → модель близька до OLS, ризик overfitting (високий Train R², низький Val R²).
    - Дуже великий α → сильна регуляризація, underfitting (обидва R² низькі).
    - Оптимум — де Val R² максимальний і розрив з Train R² невеликий.
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("mdl", model_cls())])
    train_scores, val_scores = validation_curve(
        pipe, X_all, y_all,
        param_name=f"mdl__{param_name}",
        param_range=param_range,
        cv=cv, scoring="r2", n_jobs=None
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.semilogx(param_range, train_mean, marker='o', label='Train R²')
    plt.semilogx(param_range, val_mean, marker='o', label='Val R²')
    plt.xlabel(param_name)
    plt.ylabel("R² (вище — краще)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_validation_curve(Ridge, "alpha", alphas_vc, X, y, "Validation Curve — Ridge (α vs R²), target: alcohol", cv=5)
plot_validation_curve(Lasso, "alpha", alphas_vc, X, y, "Validation Curve — Lasso (α vs R²), target: alcohol", cv=5)
