import numpy as np

# ── Bölüm 2.1 ──────────────────────────────────────────────
print("=== Bölüm 2.1: Kanser Teşhisi ===")

L_21 = np.array([[0, 1],
                  [1000, 0]])          # L[karar, gerçek]

posterior_21 = np.array([0.02, 0.98]) # [P(C1|x), P(C2|x)]

for i, karar in enumerate(['C1 (Kanser)', 'C2 (Sağlıklı)']):
    R = np.dot(L_21[i], posterior_21)
    print(f"  R({karar}|x) = {R:.2f}")

en_iyi = np.argmin([np.dot(L_21[i], posterior_21) for i in range(2)])
print(f"\n  → Minimum kayıplı karar: C{en_iyi+1}")

# ── Bölüm 2.2 ──────────────────────────────────────────────
print("\n=== Bölüm 2.2: Üç Sınıflı Problem ===")

L_22 = np.array([[0,  1,  2],
                  [4,  0,  1],
                  [10, 3,  0]])

posterior_22 = np.array([0.7, 0.2, 0.1])

R = L_22 @ posterior_22              # matris-vektör çarpımı
for i, r in enumerate(R):
    print(f"  R(C{i+1}|x) = {r:.2f}")

en_iyi_kayip  = np.argmin(R)
en_iyi_poster = np.argmax(posterior_22)

print(f"\n  → Min beklenen kayıp kararı : C{en_iyi_kayip+1}  (R={R[en_iyi_kayip]:.2f})")
print(f"  → Max posterior kararı      : C{en_iyi_poster+1}  (P={posterior_22[en_iyi_poster]:.2f})")
print(f"\n  Bu problemde iki yöntem {'aynı' if en_iyi_kayip==en_iyi_poster else 'farklı'} kararı veriyor.")