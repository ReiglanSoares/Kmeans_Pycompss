import time
import logging
import numpy as np
from pycompss.api.api import compss_wait_on, compss_barrier
from apps import kmeans_fragment, reduce_and_update


# === PARÂMETROS ===
N_POINTS = 131_072_000     
#N_POINTS = 2_072_000
DIMENSIONS = 100
K = 1000
N_FRAGMENTS = 1024
ITERATIONS = 10
SEED = 42


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
def main():
    setup_logging()

    logging.info("========== KMEANS – MODELO COMPSs (ARTIGO) ==========")
    logging.info(f"Pontos totais : {N_POINTS}")
    logging.info(f"Dimensões    : {DIMENSIONS}")
    logging.info(f"Clusters (K) : {K}")
    logging.info(f"Fragmentos   : {N_FRAGMENTS}")
    logging.info(f"Iterações    : {ITERATIONS}")
    logging.info("====================================================")

    points_per_fragment = N_POINTS // N_FRAGMENTS

    np.random.seed(SEED)
    centroids = np.random.random((K, DIMENSIONS)).astype(np.float64)

    start_total = time.time()

    for it in range(ITERATIONS):
        logging.info(f"--- ITERAÇÃO {it + 1}/{ITERATIONS} ---")
        iter_start = time.time()

        partials = []

        for frag_id in range(N_FRAGMENTS):
            partials.append(
                kmeans_fragment(
                    SEED + frag_id + it * 100_000,
                    points_per_fragment,
                    DIMENSIONS,
                    centroids
                )
            )

        centroids = reduce_and_update(centroids, partials)
        # Sincroniza iteração
        centroids = compss_wait_on(centroids)
        compss_barrier()

        logging.info(
            f"[MAIN] Iteração {it + 1} concluída em "
            f"{time.time() - iter_start:.2f} s"
        )

    logging.info("========== FIM ==========")
    logging.info(
        f"Tempo total (10 iterações): "
    )
    logging.info("========== FIM ==========")
    logging.info(
        f"Tempo total (10 iterações): "
        f"{time.time() - start_total:.2f} s"
    )
  
if __name__ == "__main__":
    main()
