import numpy as np

np.random.seed(42)

# Kutu içerikleri
kutular = {
    'r': {'prob': 0.2, 'meyveler': ['elma']*3 + ['portakal']*4 + ['limon']*3},
    'b': {'prob': 0.2, 'meyveler': ['elma']*1 + ['portakal']*1},
    'g': {'prob': 0.6, 'meyveler': ['elma']*3 + ['portakal']*3 + ['limon']*4},
}

def simulasyon(N):
    elma_sayisi = 0
    portakal_sayisi = 0
    portakal_yesil = 0

    for _ in range(N):
        # Kutu seç
        kutu = np.random.choice(['r', 'b', 'g'], p=[0.2, 0.2, 0.6])
        # Meyve seç
        meyve = np.random.choice(kutular[kutu]['meyveler'])

        if meyve == 'elma':
            elma_sayisi += 1
        if meyve == 'portakal':
            portakal_sayisi += 1
            if kutu == 'g':
                portakal_yesil += 1

    p_elma_sim = elma_sayisi / N
    p_g_portakal_sim = portakal_yesil / portakal_sayisi

    return p_elma_sim, p_g_portakal_sim

# Teorik değerler
P_ELMA     = 0.34
P_G_PORTAKAL = 0.50

for N in [100_000, 1_000_000]:
    p_elma, p_gp = simulasyon(N)
    print(f"\nN = {N:,}")
    print(f"  P̂(elma)       = {p_elma:.4f}  | Teorik: {P_ELMA:.4f} | Hata: {abs(p_elma - P_ELMA):.5f}")
    print(f"  P̂(g|portakal) = {p_gp:.4f}  | Teorik: {P_G_PORTAKAL:.4f} | Hata: {abs(p_gp - P_G_PORTAKAL):.5f}")