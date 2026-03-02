#!/bin/bash

# Configuration Paths
ORCHESTRATOR_CONF="config/pipeline_orchestrated.json"
SINGLE_LOOP_CONF="config/pipeline_single_loop.json"

echo "======================================================="
echo "Starting 10 Iterations of Experiment Suite"
echo "Orchestrator Config: $ORCHESTRATOR_CONF"
echo "Single Loop Config:  $SINGLE_LOOP_CONF"
echo "======================================================="

for i in {1..10}
do
    echo ""
    echo "-------------------------------------------------------"
    echo "ITERATION $i / 10"
    echo "-------------------------------------------------------"
    
    # Run the Manager-Specialist Architecture
    echo "[$(date +'%T')] Starting Orchestrator Pipeline..."
    uv run main.py "$ORCHESTRATOR_CONF" --mode orchestrator
    
    # simple sleep to separate logs visually or let VRAM settle fully
    sleep 2

    # Run the Single-Loop Architecture
    echo "[$(date +'%T')] Starting Single-Loop Pipeline..."
    uv run main.py "$SINGLE_LOOP_CONF" --mode single-loop

    echo "[$(date +'%T')] Iteration $i complete."
done

echo ""
echo "======================================================="
echo "All 10 iterations finished successfully."
echo "======================================================="
