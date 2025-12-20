import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. Chargement & préparation
# ============================
image_nom = "7128.jpg"  

# Lire en niveaux de gris
image_gray = cv2.imread(image_nom, cv2.IMREAD_GRAYSCALE)
if image_gray is None:
    raise RuntimeError("Impossible de lire l'image.")

m, n = image_gray.shape
print(f"Image grise : {n}x{m}")

# Normaliser en [0,1]
V_true = image_gray.astype(np.float64) / 255.0

# ============================
# 2. Création du masque M
# ============================


# proportion de pixels à masquer
p_missing = 0.65   # % de points retiré 

rng = np.random.default_rng(42)
M = (rng.random(size=(m, n)) > p_missing).astype(np.float64)
# M=1 : pixel gardé, M=0 : pixel effacé

V_obs = M * V_true  # pixels manquants mis à 0

# ============================
# 3. NMF avec masque
# ============================

def masked_nmf(V, M, r, max_iter=200, eps=1e-9, random_state=0):
    """
    NMF avec masque binaire M (1 = observé, 0 = manquant).
    Minimise || M ⊙ (V - WH) ||_F^2
    V, M : (m x n)
    r : rang
    """
    m, n = V.shape
    rng = np.random.default_rng(random_state)

    # Initialisation positive aléatoire
    W = rng.random((m, r))
    H = rng.random((r, n))

    for it in range(max_iter):
        # WH
        WH = W @ H

        # Masquage
        WH_masked = M * WH

        # --- Mise à jour de H ---
        num_H = W.T @ (M * V)            # (r x n)
        den_H = W.T @ WH_masked + eps    # (r x n)
        H *= num_H / den_H

        # Recalculer WH après maj de H
        WH = W @ H
        WH_masked = M * WH

        # --- Mise à jour de W ---
        num_W = (M * V) @ H.T            # (m x r)
        den_W = WH_masked @ H.T + eps    # (m x r)
        W *= num_W / den_W

    return W, H


# Paramètres NMF
r = 30       
max_iter = 300

print("Lancement de la NMF masquée...")
W, H = masked_nmf(V_true, M, r=r, max_iter=max_iter, random_state=42)
V_recon = W @ H
V_recon = np.clip(V_recon, 0.0, 1.0)

# ============================
# 4. Évaluation & affichage
# ============================

# Erreur globale
err_full = np.linalg.norm(V_true - V_recon, "fro")

# Erreur sur les pixels manquants seulement
missing_mask = 1.0 - M
num_missing = missing_mask.sum()
if num_missing > 0:
    err_missing = np.linalg.norm(missing_mask * (V_true - V_recon), "fro") / np.sqrt(num_missing)
else:
    err_missing = 0.0

rmse_known   = np.linalg.norm(M * (V_true - V_recon), "fro") / np.sqrt(m*n - num_missing)
rmse_global = err_full / np.sqrt(m * n)

print(f"Erreur Frobenius globale   : {err_full:.4f}")
print(f"RMSE sur pixels manquants   : {err_missing:.4f}")
print(f"RMSE sur pixels connus  : {rmse_known:.4f}")
print(f"RMSE globale    : {rmse_global:.4f}")

plt.figure(figsize=(24, 8))

plt.subplot(1, 3, 1)
plt.imshow(V_true, cmap="gray")
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(V_obs, cmap="gray")
plt.title("Image avec pixels masqués")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(V_recon, cmap="gray")
plt.title(f"Reconstruction NNMF (r = {r})")
plt.axis("off")

plt.tight_layout()
plt.show()

