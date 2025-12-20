# compare_methods.py
import time
import matplotlib.pyplot as plt

from nmf.nmf_reconstruction import evaluate_nmf_directory
from svd.svd_reconstruction import evaluate_svd_directory
from sgd.sgd_reconstruction import evaluate_sgd_directory


def main():
    image_dir = "landscapeImages/first500"

    r_values = [10, 20, 30]

    # Pour chaque méthode, on garde 4 listes (une par type d'erreur)
    nmf_rmse_g, nmf_rmse_k, nmf_rmse_m, nmf_fro = [], [], [], []
    svd_rmse_g, svd_rmse_k, svd_rmse_m, svd_fro = [], [], [], []
    sgd_rmse_g, sgd_rmse_k, sgd_rmse_m, sgd_fro = [], [], [], []

    # Et les temps d'exécution (en secondes)
    nmf_times, svd_times, sgd_times = [], [], []

    for r in r_values:
        print(f"\n===== r = {r} =====")

        # ---------- NMF ----------
        print("➡ NMF ...")
        t0 = time.perf_counter()
        g_nmf, k_nmf, m_nmf, fro_nmf = evaluate_nmf_directory(
            image_dir,
            r,
            p_missing=0.75,
            max_iter=300,
            random_state_mask=0,
        )
        t_nmf = time.perf_counter() - t0
        nmf_times.append(t_nmf)

        nmf_rmse_g.append(g_nmf)
        nmf_rmse_k.append(k_nmf)
        nmf_rmse_m.append(m_nmf)
        nmf_fro.append(fro_nmf)
        print(f"NMF - RMSE globale   = {g_nmf:.4f}")
        print(f"NMF - RMSE connus    = {k_nmf:.4f}")
        print(f"NMF - RMSE manquants = {m_nmf:.4f}")
        print(f"NMF - Frobenius glob = {fro_nmf:.4f}")
        print(f"NMF - Temps          = {t_nmf:.3f} s")

        # ---------- SVD ----------
        print("➡ SVD ...")
        t0 = time.perf_counter()
        g_svd, k_svd, m_svd, fro_svd = evaluate_svd_directory(
            image_dir,
            r,
            p_missing=0.75,
            max_iter=50,
            tol=1e-4,
            random_state_mask=0,
        )
        t_svd = time.perf_counter() - t0
        svd_times.append(t_svd)

        svd_rmse_g.append(g_svd)
        svd_rmse_k.append(k_svd)
        svd_rmse_m.append(m_svd)
        svd_fro.append(fro_svd)
        print(f"SVD - RMSE globale   = {g_svd:.4f}")
        print(f"SVD - RMSE connus    = {k_svd:.4f}")
        print(f"SVD - RMSE manquants = {m_svd:.4f}")
        print(f"SVD - Frobenius glob = {fro_svd:.4f}")
        print(f"SVD - Temps          = {t_svd:.3f} s")

        # ---------- SGD ----------
        print("➡ SGD ...")
        t0 = time.perf_counter()
        g_sgd, k_sgd, m_sgd, fro_sgd = evaluate_sgd_directory(
            image_dir,
            r,
            p_missing=0.75,
            lr=0.03,
            n_epochs=20,
            samples_per_epoch=None,
            random_state_mask=0,
            random_state_sgd=0,
        )
        t_sgd = time.perf_counter() - t0
        sgd_times.append(t_sgd)

        sgd_rmse_g.append(g_sgd)
        sgd_rmse_k.append(k_sgd)
        sgd_rmse_m.append(m_sgd)
        sgd_fro.append(fro_sgd)
        print(f"SGD - RMSE globale   = {g_sgd:.4f}")
        print(f"SGD - RMSE connus    = {k_sgd:.4f}")
        print(f"SGD - RMSE manquants = {m_sgd:.4f}")
        print(f"SGD - Frobenius glob = {fro_sgd:.4f}")
        print(f"SGD - Temps          = {t_sgd:.3f} s")

    # ===== Affichage récap des temps (sans plot) =====
    print("\n===== Récapitulatif des temps d'exécution (en secondes) =====")
    print("r\tNMF\t\tSVD\t\tSGD")
    for i, r in enumerate(r_values):
        print(f"{r}\t{nmf_times[i]:.3f}\t\t{svd_times[i]:.3f}\t\t{sgd_times[i]:.3f}")

    # ===== Plot des 4 types d'erreur sur UNE figure =====
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1) RMSE globale
    ax1.plot(r_values, nmf_rmse_g, marker="o", label="NMF")
    ax1.plot(r_values, svd_rmse_g, marker="o", label="SVD")
    ax1.plot(r_values, sgd_rmse_g, marker="o", label="SGD")
    ax1.set_xlabel("Rang r")
    ax1.set_ylabel("RMSE globale")
    ax1.set_title("RMSE globale vs r")
    ax1.grid(True)
    ax1.legend()

    # 2) RMSE sur pixels connus
    ax2.plot(r_values, nmf_rmse_k, marker="o", label="NMF")
    ax2.plot(r_values, svd_rmse_k, marker="o", label="SVD")
    ax2.plot(r_values, sgd_rmse_k, marker="o", label="SGD")
    ax2.set_xlabel("Rang r")
    ax2.set_ylabel("RMSE connus")
    ax2.set_title("RMSE sur pixels connus vs r")
    ax2.grid(True)
    ax2.legend()

    # 3) RMSE sur pixels manquants
    ax3.plot(r_values, nmf_rmse_m, marker="o", label="NMF")
    ax3.plot(r_values, svd_rmse_m, marker="o", label="SVD")
    ax3.plot(r_values, sgd_rmse_m, marker="o", label="SGD")
    ax3.set_xlabel("Rang r")
    ax3.set_ylabel("RMSE manquants")
    ax3.set_title("RMSE sur pixels manquants vs r")
    ax3.grid(True)
    ax3.legend()

    # 4) Norme de Frobenius globale
    ax4.plot(r_values, nmf_fro, marker="o", label="NMF")
    ax4.plot(r_values, svd_fro, marker="o", label="SVD")
    ax4.plot(r_values, sgd_fro, marker="o", label="SGD")
    ax4.set_xlabel("Rang r")
    ax4.set_ylabel("‖V - V_recon‖_F")
    ax4.set_title("Erreur Frobenius globale vs r")
    ax4.grid(True)
    ax4.legend()

    fig.suptitle("Comparaison NMF / SVD / SGD pour différents rangs r", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Enregistrer la figure dans un fichier
    output_file = "comparaison_erreurs.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ Figure sauvegardée dans le fichier : {output_file}")

    # (optionnel)
    plt.show()


if __name__ == "__main__":
    main()
