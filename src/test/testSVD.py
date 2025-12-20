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
p_missing = 0.5  

rng = np.random.default_rng(42)
M = (rng.random(size=(m, n)) > p_missing).astype(np.float64)
# M = 1 : pixel gardé, M = 0 : pixel effacé

V_obs = M * V_true  # pixels manquants mis à 0

# ============================
# 3. Reconstruction par SVD
# ============================

def svd_inpainting(V_obs, M, r, max_iter=50, tol=1e-4):
    """
    Reconstruction d'une image incomplète par SVD de rang r.
    V_obs : matrice (m x n) avec pixels manquants mis à 0
    M     : masque binaire (1 = connu, 0 = manquant)
    r     : rang de l'approximation
    """
    m, n = V_obs.shape
    # Matrice initiale : on part de l'image observée
    X = V_obs.copy()

    for it in range(max_iter):
        # SVD de X
        U, s, VT = np.linalg.svd(X, full_matrices=False)  # U (m x k), s (k,), VT (k x n)

        # Troncature au rang r
        r_eff = min(r, s.size)   # au cas où r > min(m,n)
        U_r = U[:, :r_eff]       # (m x r_eff)
        s_r = s[:r_eff]          # (r_eff,)
        VT_r = VT[:r_eff, :]     # (r_eff x n)

        # Approximation de rang r : X_r = U_r diag(s_r) VT_r
        X_r = (U_r * s_r) @ VT_r   # trick : multiplie chaque colonne de U_r par s_r

        # On impose les pixels connus, et on garde X_r sur les pixels manquants
        X_new = M * V_obs + (1.0 - M) * X_r

        # Calcul de variation relative pour critère d'arrêt
        num = np.linalg.norm(X_new - X, "fro")
        den = np.linalg.norm(X, "fro") + 1e-12
        rel_change = num / den

        #print(f"Itération {it+1}, variation relative = {rel_change:.6f}")

        X = X_new

        if rel_change < tol:
            print("Convergence atteinte.")
            break

    # On sature dans [0,1] pour rester une image valide
    X = np.clip(X, 0.0, 1.0)
    return X

# Paramètres SVD
r = 50        
max_iter = 100

print("Lancement de la reconstruction par SVD...")
V_recon = svd_inpainting(V_obs, M, r=r, max_iter=max_iter)

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
plt.title(f"Reconstruction SVD (r = {r})")
plt.axis("off")

plt.tight_layout()
plt.show()
