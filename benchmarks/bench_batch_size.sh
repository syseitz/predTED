#!/usr/bin/env bash
# benchmarks/bench_batch_size.sh — Sweep BATCH_SIZE for optimal throughput.
# Usage: bash benchmarks/bench_batch_size.sh [structures_file]

set -euo pipefail

STRUCTURES="${1:-data/structures.txt}"
N_LINES=$(wc -l < "$STRUCTURES")
echo "Structures: $N_LINES from $STRUCTURES"
echo ""

BEST_TIME=999999
BEST_BS=8192

for BS in 1024 2048 4096 8192 16384 32768 65536; do
    echo -n "BATCH_SIZE=$BS ... "

    # Build with this batch size
    make cli CFLAGS="-O2 -Wall -Wno-deprecated-declarations -march=native -DBATCH_SIZE=$BS" 2>/dev/null

    # Time it (3 runs, take best)
    BEST_RUN=999999
    for run in 1 2 3; do
        T=$( { time bin/predted -u < "$STRUCTURES" > /dev/null ; } 2>&1 | grep real | awk '{print $2}' | sed 's/[ms]/ /g' | awk '{print $1*60+$2}' )
        if (( $(echo "$T < $BEST_RUN" | bc -l) )); then
            BEST_RUN=$T
        fi
    done

    echo "${BEST_RUN}s"

    if (( $(echo "$BEST_RUN < $BEST_TIME" | bc -l) )); then
        BEST_TIME=$BEST_RUN
        BEST_BS=$BS
    fi
done

echo ""
echo "=== Best: BATCH_SIZE=$BEST_BS (${BEST_TIME}s) ==="
