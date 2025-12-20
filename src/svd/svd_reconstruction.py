# svd_reconstruction.py
import os
import cv2
import numpy as np


def svd_inpainting(V_obs, M, r, max_iter=50, tol=1e-4):
    """
    Reconstruction d'une image incomplète par SVD de rang r.
    V_obs : matrice (m x n) avec pixels manquants mis à 0
    M     : masque binaire (1 = connu, 0 = manquant)
    r     : rang de l'approximation
    """
    m, n = V_obs.shape
    X = V_obs.copy()

    for it in range(max_iter):
        # SVD de X
        U, s, VT = np.linalg.svd(X, full_matrices=False)

        # Troncature au rang r
        r_eff = min(r, s.size)
        U_r = U[:, :r_eff]
        s_r = s[:r_eff]
        VT_r = VT[:r_eff, :]

        # Approximation de rang r : X_r = U_r diag(s_r) VT_r
        X_r = (U_r * s_r) @ VT_r

        # Imposer les pixels connus
        X_new = M * V_obs + (1.0 - M) * X_r

        # Critère d'arrêt
        num = np.linalg.norm(X_new - X, "fro")
        den = np.linalg.norm(X, "fro") + 1e-12
        rel_change = num / den

        # print(f"Itération {it+1}, variation relative = {rel_change:.6f}")
        X = X_new

        if rel_change < tol:
            # print("Convergence atteinte.")
            break

    X = np.clip(X, 0.0, 1.0)
    return X


def evaluate_svd_directory(
    directory,
    r,
    p_missing=0.5,
    max_iter=50,
    tol=1e-4,
    random_state_mask=0,
):
    """
    Applique la reconstruction SVD sur toutes les images grises d'un répertoire.

    Retourne (rmse_global, rmse_known, rmse_missing, fro_global).
    """
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    file_names = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(image_extensions)
    )

    if not file_names:
        raise ValueError(f"Aucune image trouvée dans le répertoire : {directory}")

    rng_mask = np.random.default_rng(random_state_mask)

    total_sse = 0.0
    total_sse_known = 0.0
    total_sse_missing = 0.0
    total_pixels = 0
    total_known = 0.0
    total_missing = 0.0

    for fname in file_names:
        path = os.path.join(directory, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Impossible de lire l'image : {path}, on passe.")
            continue

        m, n = img.shape
        V_true = img.astype(np.float64) / 255.0

        # Masque
        M = (rng_mask.random(size=(m, n)) > p_missing).astype(np.float64)
        V_obs = M * V_true  # image avec trous

        # Reconstruction SVD
        V_recon = svd_inpainting(V_obs, M, r, max_iter=max_iter, tol=tol)

        # Erreurs
        E = V_true - V_recon
        miss_mask = 1.0 - M

        sse = np.sum(E ** 2)
        sse_known = np.sum((M * E) ** 2)
        sse_missing = np.sum((miss_mask * E) ** 2)

        total_sse += sse
        total_sse_known += sse_known
        total_sse_missing += sse_missing

        total_pixels += m * n
        total_known += M.sum()
        total_missing += miss_mask.sum()

    rmse_global = np.sqrt(total_sse / total_pixels)
    rmse_known = np.sqrt(total_sse_known / total_known) if total_known > 0 else float("nan")
    rmse_missing = np.sqrt(total_sse_missing / total_missing) if total_missing > 0 else float("nan")
    fro_global = np.sqrt(total_sse)

    return rmse_global, rmse_known, rmse_missing, fro_global


if __name__ == "__main__":
    dir_path = "images"  # à adapter
    r = 20
    rmse_g, rmse_k, rmse_m, fro_g = evaluate_svd_directory(dir_path, r)
    print("SVD - RMSE globale   :", rmse_g)
    print("SVD - RMSE connus    :", rmse_k)
    print("SVD - RMSE manquants :", rmse_m)
    print("SVD - Frobenius glob :", fro_g)
