# sgd_reconstruction.py
import os
import cv2
import numpy as np


def sgd_factorization(
    V_true,
    M,
    r,
    lr=0.01,
    n_epochs=20,
    samples_per_epoch=None,
    random_state=0,
):
    """
    Factorisation V_true ≈ W H par descente de gradient stochastique
    en utilisant uniquement les pixels où M=1.
    """
    m, n = V_true.shape
    rng = np.random.default_rng(random_state)

    known_positions = np.argwhere(M == 1.0)
    N_known = known_positions.shape[0]
    if N_known == 0:
        raise ValueError("Aucun pixel connu (M=1) dans cette image.")

    if samples_per_epoch is None:
        samples_per_epoch = N_known

    # Initialisation W, H (positives)
    W = rng.random((m, r)) * 0.1
    H = rng.random((r, n)) * 0.1

    for epoch in range(n_epochs):
        idx = rng.integers(low=0, high=N_known, size=samples_per_epoch)

        for t in idx:
            i, j = known_positions[t]

            v_hat = W[i, :] @ H[:, j]
            e = v_hat - V_true[i, j]

            grad_W_i = e * H[:, j]
            grad_H_j = e * W[i, :]

            W[i, :] -= lr * grad_W_i
            H[:, j] -= lr * grad_H_j

            # Projection non-négative
            W[i, :] = np.maximum(W[i, :], 0.0)
            H[:, j] = np.maximum(H[:, j], 0.0)

        # (Optionnel) suivi de l'erreur :
        # V_recon_epoch = W @ H
        # E_known = M * (V_true - V_recon_epoch)
        # rmse_known = np.linalg.norm(E_known, "fro") / np.sqrt(N_known)
        # print(f"Époque {epoch+1}/{n_epochs} - RMSE connus = {rmse_known:.4f}")

    V_recon = W @ H
    V_recon = np.clip(V_recon, 0.0, 1.0)
    return V_recon, W, H


def evaluate_sgd_directory(
    directory,
    r,
    p_missing=0.5,
    lr=0.01,
    n_epochs=20,
    samples_per_epoch=None,
    random_state_mask=0,
    random_state_sgd=0,
):
    """
    Applique la factorisation SGD sur toutes les images grises d'un répertoire.

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

    for idx_file, fname in enumerate(file_names):
        path = os.path.join(directory, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Impossible de lire l'image : {path}, on passe.")
            continue

        m, n = img.shape
        V_true = img.astype(np.float64) / 255.0

        # Masque
        M = (rng_mask.random(size=(m, n)) > p_missing).astype(np.float64)

        # SGD factorization
        V_recon, W, H = sgd_factorization(
            V_true,
            M,
            r,
            lr=lr,
            n_epochs=n_epochs,
            samples_per_epoch=samples_per_epoch,
            random_state=random_state_sgd + idx_file,  # légère variation par image
        )

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
    rmse_g, rmse_k, rmse_m, fro_g = evaluate_sgd_directory(dir_path, r)
    print("SGD - RMSE globale   :", rmse_g)
    print("SGD - RMSE connus    :", rmse_k)
    print("SGD - RMSE manquants :", rmse_m)
    print("SGD - Frobenius glob :", fro_g)
