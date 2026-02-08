import os
import re
import psutil
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import numpy as np
import subprocess
import random

# Globale Variablen
_GLOBAL_STRUCTS: List[str] = []
local_struct_pattern = re.compile(r"^([().]+)\s+\((-?\d+\.\d+)\)\s+(\d+)\s+z=\s*(-?\d+\.\d+)$")

def load_bed_file(bed_path: str) -> Dict[str, List[Tuple[int, int]]]:
    regions: Dict[str, List[Tuple[int, int]]] = {}
    with open(bed_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            start, end = int(parts[1]), int(parts[2])
            region = parts[3]
            regions.setdefault(region, []).append((start, end))
    return regions

def is_within_region(start: int, end: int,
                     regions: Dict[str, List[Tuple[int, int]]],
                     region_filter: Optional[str]) -> bool:
    if region_filter is None:
        return True
    if region_filter not in regions:
        return False
    return any(r_start <= start <= end <= r_end
               for (r_start, r_end) in regions[region_filter])

def process_file(file_path: str,
                 bed_folder: Optional[str],
                 region_filter: Optional[str]) -> List[str]:
    transcript = os.path.splitext(os.path.basename(file_path))[0].split("_0-")[0]
    regions: Dict[str, List[Tuple[int, int]]] = {}
    if region_filter is not None and bed_folder:
        bf = os.path.join(bed_folder, f"{transcript}.bed")
        if os.path.isfile(bf):
            regions = load_bed_file(bf)
        else:
            print(f"[Warnung] BED fehlt für Transkript {transcript}")
            return []

    with open(file_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 1:
        print(f"[Warnung] Keine Strukturdaten in {file_path}")
        return []

    out: List[str] = []
    for line in lines:
        m = local_struct_pattern.match(line)
        if not m:
            continue
        struct = m.group(1)
        start_pos = int(m.group(3))
        end_pos = start_pos + len(struct) - 1
        if not is_within_region(start_pos, end_pos, regions, region_filter):
            continue
        out.append(struct)
    return out

def compute_distance(pair: Tuple[int,int]) -> Tuple[Tuple[int,int], int]:
    i, j = pair
    struct1 = _GLOBAL_STRUCTS[i]
    struct2 = _GLOBAL_STRUCTS[j]
    cmd = ['RNAdistance', '-D', 'f']
    input_data = f"{struct1}\n{struct2}\n"
    result = subprocess.run(cmd, input=input_data, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    for line in lines:
        if line.startswith('f:'):
            dist = int(float(line.split()[1]))
            return (i, j), dist
    raise ValueError("Could not parse RNAdistance output")

def print_status(step: str):
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[{step}] RAM usage: {mem_mb:.1f} MB")

def generate_training_data(input_folder: str, bed_folder: Optional[str], region_filter: Optional[str], structures_file: str = 'structures.txt', ted_matrix_file: str = 'ted_matrix.txt', num_structures: int = 1500):
    global _GLOBAL_STRUCTS
    _GLOBAL_STRUCTS.clear()

    print_status("1) Scanning .lfold files")
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".lfold")]
    print(f"Found {len(files)} files; using {cpu_count()} cores.")

    print_status("2) Extracting structures")
    all_structs = []
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(process_file, bed_folder=bed_folder, region_filter=region_filter), files)
        for result in results:
            all_structs.extend(result)
    
    if not all_structs:
        print("No structures extracted; exiting.")
        return

    print(f"Extracted {len(all_structs)} structures")
    if len(all_structs) > num_structures:
        selected_structs = random.sample(all_structs, num_structures)
    else:
        selected_structs = all_structs
    print(f"Selected {len(selected_structs)} structures")

    _GLOBAL_STRUCTS = selected_structs

    with open(structures_file, 'w') as f:
        for struct in _GLOBAL_STRUCTS:
            f.write(struct + '\n')

    m = len(_GLOBAL_STRUCTS)
    all_pairs = [(i, j) for i in range(m) for j in range(i+1, m)]
    num_pairs = len(all_pairs)
    print(f"Generated {num_pairs} pairs for TED computation")

    print_status("3) Computing pairwise TED")
    with Pool(cpu_count()) as pool:
        pair_distances = list(tqdm(pool.imap_unordered(compute_distance, all_pairs), total=num_pairs, desc="Computing TED"))

    ted_matrix = np.zeros((m, m), dtype=int)
    for (i, j), dist in pair_distances:
        ted_matrix[i, j] = dist
        ted_matrix[j, i] = dist

    np.savetxt(ted_matrix_file, ted_matrix, delimiter=' ', fmt='%d')

    print(f"Structures saved to '{structures_file}'")
    print(f"TED matrix saved to '{ted_matrix_file}'")

if __name__ == "__main__":
    generate_training_data(
        input_folder='../../data/Human/protein_coding/LFOLD_Z-2_protein_coding_strict_filtered',
        bed_folder='../../data/Human/protein_coding/BED6__protein_coding_strict',
        region_filter=None,
        structures_file='structures.txt',
        ted_matrix_file='ted_matrix.txt'
    )