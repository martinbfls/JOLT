import numpy as np


def partition_across_workers(N, idx_flipped, idx_clean, labels_final, num_honests, num_poisoned,
                              seed=0, shuffle_clean=True):
    num_workers = num_honests + num_poisoned

    if shuffle_clean:
        idx_clean = idx_clean.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(idx_clean)

    base = N // num_workers
    remainder = N % num_workers
    sizes = np.array(
        [base + (w < remainder) for w in range(num_workers)], dtype=int
    )

    flipped_split = np.array_split(idx_flipped, num_poisoned)

    worker_indices = []
    worker_labels = []

    clean_ptr = 0

    for w in range(num_honests):
        sz = sizes[w]
        sel = idx_clean[clean_ptr: clean_ptr + sz]
        clean_ptr += sz

        worker_indices.append(sel)
        worker_labels.append(labels_final[sel])

    for p in range(num_poisoned):
        w = num_honests + p
        sz = sizes[w]

        flipped_p = flipped_split[p]
        remaining = sz - len(flipped_p)
        if remaining < 0:
            raise ValueError(f"Too many flipped samples for poisoned worker {w}")

        sel_clean = idx_clean[clean_ptr: clean_ptr + remaining]
        clean_ptr += remaining

        sel = np.concatenate([sel_clean, flipped_p])

        worker_indices.append(sel)
        worker_labels.append(labels_final[sel])

    return worker_indices, worker_labels
