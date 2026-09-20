#!/bin/bash

# Usage: ./benchmark_versions.sh <video_file> [runs] [output_dir]

set -e

VIDEO_FILE="$1"
RUNS="${2:-3}"
OUTPUT_DIR="${3:-benchmark_results}"

if [ -z "$VIDEO_FILE" ]; then
    echo "Usage: $0 <video_file> [runs] [output_dir]"
    echo "Example: $0 samples/ACET.mp4 3 results"
    exit 1
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file '$VIDEO_FILE' not found!"
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
CHANGES_STASHED=0

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$CURRENT_BRANCH" ]; then
        # Only checkout if we are not already on the branch (optimization)
        if [ "$(git branch --show-current)" != "$CURRENT_BRANCH" ]; then
            echo "Returning to $CURRENT_BRANCH..."
            git checkout -f "$CURRENT_BRANCH" 2>&1 > /dev/null || true
        fi
    fi
    
    if [ "$CHANGES_STASHED" -eq 1 ]; then
        echo "Restoring local changes..."
        git stash pop || echo "Warning: Failed to pop stash. You may need to run 'git stash pop' manually."
    fi
}
trap cleanup EXIT

# Stash any local changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Stashing local changes..."
    git stash push -u -m "benchmark_versions.sh temporary stash"
    CHANGES_STASHED=1
fi

mkdir -p "$OUTPUT_DIR"

echo "GPGPU Benchmark Suite"
echo "Video: $VIDEO_FILE"
echo "Runs per version: $RUNS"
echo "Results directory: $OUTPUT_DIR"
echo ""

declare -a VERSIONS=(
    "d76e7ce|v0.1.0|BG OPTI 0, Noise OPTI 0, Threshold OPTI 0|gpu"
    "dfb697b|v0.2.0|BG OPTI 1, Noise OPTI 0, Threshold OPTI 0|gpu"
    "9302691|v0.3.0|BG OPTI 1, Noise OPTI 1, Threshold OPTI 0|gpu"
    "4c22c84|v0.4.0|BG OPTI 2, Noise OPTI 1, Threshold OPTI 0|gpu"
    "29b0bd9|v0.5.0|BG OPTI 2, Noise OPTI 1, Threshold OPTI 1|gpu"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$OUTPUT_DIR/benchmark_$TIMESTAMP.csv"
echo "Version,Description,Mode,AvgTime,MinTime,MaxTime,Runs" > "$CSV_FILE"

declare -a RESULTS=()

for VERSION_INFO in "${VERSIONS[@]}"; do
    IFS='|' read -r HASH NAME DESC MODE <<< "$VERSION_INFO"
    
    echo "================================================"
    echo "Testing: $NAME - $DESC"
    echo "================================================"
    
    echo "Checking out $HASH..."
    # Force checkout to discard changes from previous iteration
    git checkout -f "$HASH" 2>&1 > /dev/null
    
    if [ $? -ne 0 ]; then
        echo "Failed to checkout $HASH"
        continue
    fi

    # Build
    echo "Building..."
    rm -rf build
    cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 > /dev/null
    
    set +e  # Temporarily disable exit on error
    cmake --build build --config Release 2>&1 > /dev/null
    BUILD_EXIT=$?
    set -e  # Re-enable exit on error
    
    if [ $BUILD_EXIT -ne 0 ]; then
        echo "Build failed for $NAME"
        continue
    fi
    
    # Set up environment for GStreamer
    export XDG_RUNTIME_DIR=/tmp/runtime-$USER
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    
    declare -a TOTAL_TIMES=()
    declare -a KERNEL_TIMES=()
    
    echo ""
    for ((i=1; i<=RUNS; i++)); do
        echo "  Run $i/$RUNS:"
        
        START=$(date +%s.%N)
        OUTPUT=$(./build/stream --mode=$MODE "$VIDEO_FILE" --output=/dev/null 2>&1)
        EXIT_CODE=$?
        END=$(date +%s.%N)
        
        if [ $EXIT_CODE -eq 0 ]; then
            DURATION=$(awk "BEGIN {printf \"%.3f\", $END - $START}")
            TOTAL_TIMES+=($DURATION)
            
            # Extract kernel time
            # Use grep -a to handle potential binary output issues, take the LAST occurrence (most accurate average)
            K_TIME=$(echo "$OUTPUT" | grep -a "AVERAGE_COMPUTE_TIME" | tail -n 1 | awk '{print $2}')
            if [ -n "$K_TIME" ]; then
                KERNEL_TIMES+=($K_TIME)
                printf "    ├─ Total time:  %7.2f s\n" "$DURATION"
                printf "    └─ Kernel time: %7.2f ms\n" "$K_TIME"
            else
                printf "    ├─ Total time:  %7.2f s\n" "$DURATION"
                printf "    └─ Kernel time: N/A\n"
            fi
        else
            echo "    └─ Run failed!"
        fi
    done
    
    if [ ${#TOTAL_TIMES[@]} -gt 0 ]; then
        # Calculate Average Total Time
        SUM=0
        for TIME in "${TOTAL_TIMES[@]}"; do
            SUM=$(awk "BEGIN {printf \"%.3f\", $SUM + $TIME}")
        done
        AVG_TOTAL=$(awk "BEGIN {printf \"%.3f\", $SUM / ${#TOTAL_TIMES[@]}}")
        
        # Calculate Average Kernel Time
        AVG_KERNEL="N/A"
        if [ ${#KERNEL_TIMES[@]} -gt 0 ]; then
            SUM_K=0
            for TIME in "${KERNEL_TIMES[@]}"; do
                SUM_K=$(awk "BEGIN {printf \"%.3f\", $SUM_K + $TIME}")
            done
            AVG_KERNEL=$(awk "BEGIN {printf \"%.3f\", $SUM_K / ${#KERNEL_TIMES[@]}}")
        fi
        
        echo ""
        echo "  ════════════════════════════════════════"
        printf "  AVERAGES (${#TOTAL_TIMES[@]} runs):\n"
        printf "     Total time:  %7.2f s\n" "$AVG_TOTAL"
        if [ "$AVG_KERNEL" != "N/A" ]; then
            printf "     Kernel time: %7.2f ms\n" "$AVG_KERNEL"
        else
            printf "     Kernel time: N/A\n"
        fi
        echo "  ════════════════════════════════════════"
        
        echo "$NAME,$DESC,$MODE,$AVG_TOTAL,$AVG_KERNEL,${#TOTAL_TIMES[@]}" >> "$CSV_FILE"
        
        RESULTS+=("$NAME|$DESC|$MODE|$AVG_TOTAL|$AVG_KERNEL")
    fi
    
    echo ""
done



echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════════════════"
echo "                                   📊 BENCHMARK RESULTS                                      "
echo "═══════════════════════════════════════════════════════════════════════════════════════════════"
echo ""
printf "%-20s %-30s %-6s %12s %15s\n" "Version" "Description" "Mode" "Avg Total (s)" "Avg Kernel (ms)"
echo "───────────────────────────────────────────────────────────────────────────────────────────────"

BASELINE_KERNEL=""
for RESULT in "${RESULTS[@]}"; do
    IFS='|' read -r NAME DESC MODE AVG_TOTAL AVG_KERNEL <<< "$RESULT"
    if [ "$AVG_KERNEL" != "N/A" ]; then
        printf "%-20s %-30s %-6s %12.2f %15.2f\n" "$NAME" "$DESC" "$MODE" "$AVG_TOTAL" "$AVG_KERNEL"
    else
        printf "%-20s %-30s %-6s %12.2f %15s\n" "$NAME" "$DESC" "$MODE" "$AVG_TOTAL" "N/A"
    fi
    
    if [ -z "$BASELINE_KERNEL" ] && [ "$AVG_KERNEL" != "N/A" ]; then
        BASELINE_KERNEL=$AVG_KERNEL
    fi
done

echo "═══════════════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "💾 Results saved to: $CSV_FILE"

if [ ${#RESULTS[@]} -gt 1 ] && [ -n "$BASELINE_KERNEL" ]; then
    echo ""
    echo "🚀 Kernel Speedup vs baseline (${RESULTS[0]%%|*}):"
    echo ""
    
    for ((i=1; i<${#RESULTS[@]}; i++)); do
        IFS='|' read -r NAME DESC MODE AVG_TOTAL AVG_KERNEL <<< "${RESULTS[$i]}"
        if [ "$AVG_KERNEL" != "N/A" ] && [ $(awk "BEGIN {print ($AVG_KERNEL > 0)}") -eq 1 ]; then
            SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $BASELINE_KERNEL / $AVG_KERNEL}")
            printf "  %-20s: %.2fx faster\n" "$NAME" "$SPEEDUP"
        fi
    done
fi