"""Speed benchmark: predted vs RNA.tree_edit_distance() vs RNAdistance CLI."""

import time
import subprocess
from pathlib import Path

import numpy as np
import RNA

import predted


def bench_predted_matrix(structures: list) -> float:
    start = time.perf_counter()
    predted.predict_matrix(structures)
    return time.perf_counter() - start


def bench_rna_ted_python(structures: list) -> float:
    n = len(structures)
    tree_strings = [
        RNA.db_to_tree_string(s, RNA.STRUCTURE_TREE_EXPANDED)
        for s in structures
    ]
    start = time.perf_counter()
    for i in range(n):
        for j in range(i + 1, n):
            t1 = RNA.make_tree(tree_strings[i])
            t2 = RNA.make_tree(tree_strings[j])
            RNA.tree_edit_distance(t1, t2)
            RNA.free_tree(t1)
            RNA.free_tree(t2)
    return time.perf_counter() - start


def bench_rnadistance_cli(structures: list) -> float:
    input_text = "\n".join(structures) + "\n"
    start = time.perf_counter()
    result = subprocess.run(
        ["RNAdistance", "-Df", "-Xm"],
        input=input_text, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"  RNAdistance error: {result.stderr[:200]}")
    return elapsed


def main():
    structures_file = Path(__file__).parent.parent / "data" / "structures.txt"
    all_structures = structures_file.read_text().strip().splitlines()
    print(f"Loaded {len(all_structures)} structures")
    print(f"Length range: {min(len(s) for s in all_structures)}"
          f" - {max(len(s) for s in all_structures)}")

    # Warm up predted (first call loads the LightGBM model)
    _ = predted.predict(all_structures[0], all_structures[1])

    test_sizes = [10, 25, 50, 100, 250, 500]
    results = []

    for n in test_sizes:
        subset = all_structures[:n]
        n_pairs = n * (n - 1) // 2
        print(f"\n--- N = {n} ({n_pairs:,} pairs) ---")

        t_predted = bench_predted_matrix(subset)
        print(f"  predted:              {t_predted:.3f}s"
              f"  ({t_predted / n_pairs * 1000:.4f} ms/pair)")

        t_rna_py = bench_rna_ted_python(subset)
        print(f"  RNA.tree_edit_dist(): {t_rna_py:.3f}s"
              f"  ({t_rna_py / n_pairs * 1000:.4f} ms/pair)")

        t_cli = bench_rnadistance_cli(subset)
        print(f"  RNAdistance CLI:      {t_cli:.3f}s"
              f"  ({t_cli / n_pairs * 1000:.4f} ms/pair)")

        results.append({
            "n": n, "n_pairs": n_pairs,
            "predted_s": t_predted,
            "rna_python_s": t_rna_py,
            "rnadistance_cli_s": t_cli,
        })

    # Summary table
    header = (f"{'N':>5} {'Pairs':>8} | "
              f"{'predted':>9} {'RNA Py':>9} {'CLI':>9} | "
              f"{'vs RNA':>8} {'vs CLI':>8}")
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        sp_rna = (r["rna_python_s"] / r["predted_s"]
                  if r["predted_s"] > 0 else float("inf"))
        sp_cli = (r["rnadistance_cli_s"] / r["predted_s"]
                  if r["predted_s"] > 0 else float("inf"))
        print(
            f"{r['n']:>5} {r['n_pairs']:>8,} | "
            f"{r['predted_s']:>8.3f}s {r['rna_python_s']:>8.3f}s "
            f"{r['rnadistance_cli_s']:>8.3f}s | "
            f"{sp_rna:>7.1f}x {sp_cli:>7.1f}x"
        )

    # Single-pair timing
    print("\n--- Single-pair timing (median of 1000 repeats) ---")
    s1, s2 = all_structures[0], all_structures[1]
    n_repeats = 1000

    times_predted = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predted.predict(s1, s2)
        times_predted.append(time.perf_counter() - t0)

    t1_str = RNA.db_to_tree_string(s1, RNA.STRUCTURE_TREE_EXPANDED)
    t2_str = RNA.db_to_tree_string(s2, RNA.STRUCTURE_TREE_EXPANDED)
    times_rna = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        t1 = RNA.make_tree(t1_str)
        t2 = RNA.make_tree(t2_str)
        RNA.tree_edit_distance(t1, t2)
        RNA.free_tree(t1)
        RNA.free_tree(t2)
        times_rna.append(time.perf_counter() - t0)

    med_p = np.median(times_predted) * 1000
    med_r = np.median(times_rna) * 1000
    direction = "slower" if med_p > med_r else "faster"
    print(f"  predted:              {med_p:.3f} ms")
    print(f"  RNA.tree_edit_dist(): {med_r:.3f} ms")
    print(f"  Ratio:                predted is {med_p / med_r:.1f}x {direction}")


if __name__ == "__main__":
    main()
