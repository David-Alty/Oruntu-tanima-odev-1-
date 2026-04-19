import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve

# --- Parametreler ---
mu1, sig1 = 0, 1
mu2, sig2 = 2, 2   # sigma=2 → sigma²=4

def coz(p1, p2):
    # Likelihood fonksiyonları
    def f(x):
        return norm.pdf(x, mu1, sig1)*p1 - norm.pdf(x, mu2, sig2)*p2

    # İki başlangıç noktasından çöz
    s1 = fsolve(f, 1.5)[0]
    s2 = fsolve(f, -3.0)[0]
    return sorted([s1, s2])

# --- (b) Likelihood grafiği ---
x = np.linspace(0, 5, 500)
for (p1, p2), ls in [((0.6,0.4),'-'), ((0.5,0.5),'--'), ((0.4,0.6),':')]:
    plt.figure(figsize=(8,4))
    plt.plot(x, norm.pdf(x, mu1, sig1), 'b-',  label='p(x|C₁)=N(0,1)')
    plt.plot(x, norm.pdf(x, mu2, sig2), 'r-',  label='p(x|C₂)=N(2,4)')
    sinirlar = coz(p1, p2)
    for s in sinirlar:
        plt.axvline(s, color='g', linestyle='--', label=f'Karar sınırı x≈{s:.3f}')
    plt.title(f'Likelihood  |  p(C₁)={p1}, p(C₂)={p2}')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# --- (c) Posterior grafiği ---
x = np.linspace(0, 5, 500)
for p1, p2 in [(0.6,0.4),(0.5,0.5),(0.4,0.6)]:
    l1 = norm.pdf(x, mu1, sig1)*p1
    l2 = norm.pdf(x, mu2, sig2)*p2
    px = l1 + l2
    plt.figure(figsize=(8,4))
    plt.plot(x, l1/px, 'b-', label='P(C₁|x)')
    plt.plot(x, l2/px, 'r-', label='P(C₂|x)')
    for s in coz(p1, p2):
        plt.axvline(s, color='g', linestyle='--', label=f'x≈{s:.3f}')
    plt.title(f'Posterior  |  p(C₁)={p1}, p(C₂)={p2}')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# --- (d) p(x) dağılımı ---
x = np.linspace(-5, 5, 500)
p1, p2 = 0.6, 0.4
px = norm.pdf(x,mu1,sig1)*p1 + norm.pdf(x,mu2,sig2)*p2
plt.figure(figsize=(8,4))
plt.plot(x, px, color='purple', label='p(x)')
plt.title('Marjinal dağılım p(x)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()