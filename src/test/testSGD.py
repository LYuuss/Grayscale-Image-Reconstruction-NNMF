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
p_missing = 0.5   # 50% par exemple

rng = np.random.default_rng(42)
M = (rng.random(size=(m, n)) > p_missing).astype(np.float64)
# M = 1 : pixel gardé, M = 0 : pixel effacé

V_obs = M * V_true  # pixels manquants mis à 0 (juste pour affichage)

# ============================
# 3. Factorisation par SGD
# ============================

def sgd_factorization(V_true, M, r, lr=0.01, n_epochs=20,
                      samples_per_epoch=None, random_state=0):
    """
    Factorisation V_true ≈ W H par descente de gradient stochastique,
    en utilisant uniquement les pixels où M=1.

    V_true : matrice (m x n), normalisée dans [0,1]
    M      : masque binaire (1 = pixel connu, 0 = manquant)
    r      : rang latent
    lr     : learning rate (taux d'apprentissage)
    n_epochs : nombre d'époques (passes sur les données)
    samples_per_epoch : nombre d'échantillons (pixels) par époque
                        (par défaut : nb de pixels connus)
    """
    m, n = V_true.shape
    rng = np.random.default_rng(random_state)

    # Indices des pixels connus (M=1)
    known_positions = np.argwhere(M == 1.0)  # shape (N_known, 2)
    N_known = known_positions.shape[0]
    if samples_per_epoch is None:
        samples_per_epoch = N_known  # ~1 "epoch" = une passe

    # Initialisation W, H (positives)
    W = rng.random((m, r)) * 0.1
    H = rng.random((r, n)) * 0.1

    for epoch in range(n_epochs):
        # Tirage aléatoire d'indices dans les pixels connus
        idx = rng.integers(low=0, high=N_known, size=samples_per_epoch)

        for t in idx:
            i, j = known_positions[t]

            # Prédiction pour ce pixel
            v_hat = W[i, :] @ H[:, j]
            e = v_hat - V_true[i, j]  # erreur (WH - V)

            # Gradients (pixel-wise)
            grad_W_i = e * H[:, j]      # taille (r,)
            grad_H_j = e * W[i, :]      # taille (r,)

            # Mises à jour SGD
            W[i, :] -= lr * grad_W_i
            H[:, j] -= lr * grad_H_j

            # Projection non-négative
            W[i, :] = np.maximum(W[i, :], 0.0)
            H[:, j] = np.maximum(H[:, j], 0.0)

        # Suivi : RMSE sur les pixels connus à la fin de l'époque
        V_recon_epoch = W @ H
        E_known = M * (V_true - V_recon_epoch)
        rmse_known = np.linalg.norm(E_known, "fro") / np.sqrt(N_known)
        #print(f"Époque {epoch+1}/{n_epochs} - RMSE sur connus = {rmse_known:.4f}")

    V_recon = W @ H
    V_recon = np.clip(V_recon, 0.0, 1.0)
    return V_recon, W, H


# Paramètres SGD
r = 20          # rang latent
lr = 0.05       # learning rate
n_epochs = 50   # nombre de passes

print("Lancement de la factorisation par SGD...")
V_recon, W_sgd, H_sgd = sgd_factorization(
    V_true, M, r=r, lr=lr, n_epochs=n_epochs, samples_per_epoch=None, random_state=42
)

# ============================
# 4. Évaluation & affichage
# ============================

E = V_true - V_recon

N_total   = m * n
N_known   = M.sum()
N_missing = (1.0 - M).sum()

err_full = np.linalg.norm(V_true - V_recon, "fro")
rmse_global  = np.linalg.norm(E, "fro") / np.sqrt(N_total)
rmse_known   = np.linalg.norm(M * E, "fro") / np.sqrt(N_known)
rmse_missing = np.linalg.norm((1.0 - M) * E, "fro") / np.sqrt(N_missing)

print(f"Erreur Frobenius globale   : {err_full:.4f}")
print(f"RMSE sur pixels manquants   : {rmse_missing:.4f}")
print(f"RMSE sur pixels connus   : {rmse_known:.4f}")
print(f"RMSE globale    : {rmse_global:.4f}")

plt.figure(figsize=(12, 4))

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
plt.title(f"Reconstruction SGD (r = {r})")
plt.axis("off")

plt.tight_layout()
plt.show()
