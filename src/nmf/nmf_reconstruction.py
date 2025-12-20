# nmf_reconstruction.py
import os
import cv2
import numpy as np


def masked_nmf(V, M, r, max_iter=300, eps=1e-9, random_state=None):
    """
    NMF avec masque binaire M (1 = observé, 0 = manquant).
    Minimise || M ⊙ (V - WH) ||_F^2 par mises à jour multiplicatives.

    V, M : matrices (m x n)
    r    : rang (nombre de composantes)
    """
    m, n = V.shape
    rng = np.random.default_rng(random_state)

    # Initialisation positive
    W = rng.random((m, r))
    H = rng.random((r, n))

    for it in range(max_iter):
        # Reconstruction courante
        WH = W @ H

        # Masquage
        WH_masked = M * WH

        # --- Mise à jour de H ---
        num_H = W.T @ (M * V)           # (r x n)
        den_H = W.T @ WH_masked + eps   # (r x n)
        H *= num_H / den_H

        # Recalculer WH après mise à jour de H
        WH = W @ H
        WH_masked = M * WH

        # --- Mise à jour de W ---
        num_W = (M * V) @ H.T           # (m x r)
        den_W = WH_masked @ H.T + eps   # (m x r)
        W *= num_W / den_W

        # (Optionnel) tu peux ajouter un print d'erreur tous les X itérations

    return W, H


def evaluate_nmf_directory(
    directory,
    r,
    p_missing=0.5,
    max_iter=300,
    random_state_mask=0,
):
    """
    Applique la reconstruction NMF masquée sur toutes les images grises d'un répertoire.

    Paramètres
    ---------
    directory : str
        Chemin du répertoire contenant uniquement des images grayscale.
    r : int
        Rang de la factorisation W (m x r), H (r x n).
    p_missing : float
        Proportion de pixels à masquer aléatoirement.
    max_iter : int
        Nombre maximal d'itérations de l'algorithme NMF.
    random_state_mask : int
        Graine pour le masque de pixels manquants (pour reproductibilité).

    Retourne
    --------
    (rmse_global, rmse_known, rmse_missing, fro_global)
    sur l'ensemble des images du répertoire.
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

        # Masque de pixels observés
        M = (rng_mask.random(size=(m, n)) > p_missing).astype(np.float64)

        # NMF masquée
        W, H = masked_nmf(V_true, M, r, max_iter=max_iter)
        V_recon = W @ H
        V_recon = np.clip(V_recon, 0.0, 1.0)

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
    # Exemple d'utilisation
    dir_path = "images" 
    r = 20
    rmse_g, rmse_k, rmse_m, fro_g = evaluate_nmf_directory(dir_path, r)
    print("NMF - RMSE globale   :", rmse_g)
    print("NMF - RMSE connus    :", rmse_k)
    print("NMF - RMSE manquants :", rmse_m)
    print("NMF - Frobenius glob :", fro_g)
