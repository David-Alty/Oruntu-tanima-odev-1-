import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve

# --- Parametreler [cite: 38, 42, 44] ---
mu1, sig1 = 0, 1
mu2, sig2 = 2, 2   # sigma=2 → sigma²=4 [cite: 44]

def coz(p1, p2):
    # Karar sınırı bulma fonksiyonu: p(x|C1)P(C1) = p(x|C2)P(C2) [cite: 47]
    def f(x):
        return norm.pdf(x, mu1, sig1)*p1 - norm.pdf(x, mu2, sig2)*p2
    
    # İki başlangıç noktasından kök bulma (Gauss kesişimleri genelde iki tanedir)
    s1 = fsolve(f, 1.5)[0]
    s2 = fsolve(f, -3.0)[0]
    return sorted([s1, s2])

# ---------------------------------------------------------
# ŞIK (b), (e) ve (f): Likelihood (Olabilirlik) Grafikleri
# ---------------------------------------------------------
# Bu bölüm sınıfların dağılımını ve karar sınırının (dikey çizgi) 
# önsel olasılığa göre nasıl kaydığını gösterir[cite: 48, 49, 54, 55].

x_b = np.linspace(0, 5, 500)
senaryolar = [
    ((0.6, 0.4), "Şık (b): p(C1)=0.6, p(C2)=0.4"),
    ((0.5, 0.5), "Şık (e): p(C1)=0.5, p(C2)=0.5"),
    ((0.4, 0.6), "Şık (f): p(C1)=0.4, p(C2)=0.6")
]

for (p1, p2), baslik in senaryolar:
    plt.figure(figsize=(8, 4))
    plt.plot(x_b, norm.pdf(x_b, mu1, sig1), 'b-', label='p(x|C1)=N(0,1)')
    plt.plot(x_b, norm.pdf(x_b, mu2, sig2), 'r-', label='p(x|C2)=N(2,4)')
    
    sinirlar = coz(p1, p2)
    for s in sinirlar:
        if 0 <= s <= 5: # Grafikte görünür aralıktaki sınırı çiz
            plt.axvline(s, color='g', linestyle='--', label=f'Karar Sınırı x≈{s:.3f}')
            
    plt.title(f'Likelihood | {baslik}')
    plt.xlabel('x'); plt.ylabel('p(x|C)'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------
# ŞIK (c), (e) ve (f): Posterior (Sonsal) Grafikleri
# ---------------------------------------------------------
# Bu bölüm bir x değeri verildiğinde o verinin hangi sınıfa ait 
# olma olasılığını (0 ile 1 arası) gösterir[cite: 50, 51, 52].

for (p1, p2), baslik in senaryolar:
    l1 = norm.pdf(x_b, mu1, sig1) * p1
    l2 = norm.pdf(x_b, mu2, sig2) * p2
    px = l1 + l2
    
    plt.figure(figsize=(8, 4))
    plt.plot(x_b, l1/px, 'b-', label='P(C1|x) [Sonsal]')
    plt.plot(x_b, l2/px, 'r-', label='P(C2|x) [Sonsal]')
    
    for s in coz(p1, p2):
        if 0 <= s <= 5:
            plt.axvline(s, color='g', linestyle='--', label=f'Eşik x≈{s:.3f}')
            
    plt.title(f'Posterior | {baslik}')
    plt.xlabel('x'); plt.ylabel('P(C|x)'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------
# ŞIK (d): p(x) Marjinal Dağılım Grafiği
# ---------------------------------------------------------
# Bu bölüm verinin toplam yoğunluğunu (evidence) gösterir.

x_d = np.linspace(-5, 5, 500)
p1_d, p2_d = 0.6, 0.4
px_d = norm.pdf(x_d, mu1, sig1)*p1_d + norm.pdf(x_d, mu2, sig2)*p2_d

plt.figure(figsize=(8, 4))
plt.plot(x_d, px_d, color='purple', linewidth=2, label='p(x) = ∑ p(x|Ck)P(Ck)')
plt.fill_between(x_d, px_d, color='purple', alpha=0.1)
plt.title('Şık (d): Marjinal Dağılım p(x) (Evidence)')
plt.xlabel('x'); plt.ylabel('p(x)'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()