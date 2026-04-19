import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Eğitim verisi ──────────────────────────────────────
N_train = 10
x_train = np.linspace(0, 1, N_train)
epsilon  = np.random.normal(0, 0.2, N_train)          # gürültü σ=0.2
t_train  = np.sin(2 * np.pi * x_train) + epsilon

# ── 2. Tasarım matrisi Φ (derece 2 → [1, x, x²]) ─────────
def design_matrix(x, degree=2):
    return np.column_stack([x**k for k in range(degree + 1)])

Phi = design_matrix(x_train, degree=2)   # (10, 3)

# ── 3. MLE ağırlıkları: w_ML = (ΦᵀΦ)⁻¹ Φᵀ t ────────────
w_ML = np.linalg.solve(Phi.T @ Phi, Phi.T @ t_train)
print("w_ML =", np.round(w_ML, 4))       # [w0, w1, w2]

# ── 4. Hassasiyet parametresi β_ML ───────────────────────
t_pred_train = Phi @ w_ML                # eğitim tahminleri
residuals    = t_train - t_pred_train
sigma2_ML    = np.mean(residuals**2)     # 1/N Σ(tₙ - y(xₙ))²
beta_ML      = 1.0 / sigma2_ML
print(f"σ²_ML = {sigma2_ML:.4f},  β_ML = {beta_ML:.4f}")

# ── 5. Test kümesi (100 nokta) ────────────────────────────
x_test = np.linspace(0, 1, 100)
Phi_test = design_matrix(x_test, degree=2)
t_test   = Phi_test @ w_ML              # model tahminleri

# ── 6. Grafik ─────────────────────────────────────────────
x_true = np.linspace(0, 1, 300)
y_true = np.sin(2 * np.pi * x_true)

plt.figure(figsize=(9, 5))
plt.plot(x_true,  y_true,       'k--',  lw=1.5, label='Gerçek: sin(2πx)',  alpha=0.5)
plt.plot(x_test,  t_test,       'b-',   lw=2,   label='Model y(x, w_ML)')
plt.scatter(x_test, t_test,     c='teal', s=12, alpha=0.6, label='Test tahminleri')
plt.scatter(x_train, t_train,   c='red',  s=60, zorder=5,  label='Eğitim verisi')
plt.xlabel('x');  plt.ylabel('t')
plt.title(f'MLE Eğri Uydurma (derece 2) | β_ML = {beta_ML:.2f}')
plt.legend();  plt.grid(alpha=0.3);  plt.tight_layout()
plt.show()