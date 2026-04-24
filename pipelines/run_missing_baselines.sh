#!/bin/bash

# Setup environment
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Ensure we use the correct python (the one with numpy/torch)
PYTHON_EXE=$(which python)
echo "Using python: $PYTHON_EXE"

echo "Starting missing baselines run at $(date)"

# 1. CICIDS2017: DeepSVDD + TranAD (already handles mixed labels in load_cicids_windowed)
echo "Running CICIDS2017 DeepSVDD and TranAD..."
$PYTHON_EXE baselines/window_baselines.py \
  --dataset cicids2017 --data-dir dataset/CICIDS2017 \
  --train-files Tuesday-WorkingHours.pcap_ISCX.csv Wednesday-workingHours.pcap_ISCX.csv Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv \
  --test-files Friday-WorkingHours-Morning.pcap_ISCX.csv Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \
  --methods deepsvdd tranad --epochs 10 \
  --output-dir results/sota_deepsvdd_tranad_cicids \
  --out-log results/logs/deepsvdd_tranad_cicids.log

# 2. SWaT: RF + MLP (use supervised-mixed-split to see both classes)
echo "Running SWaT RF and MLP (Supervised)..."
$PYTHON_EXE baselines/window_baselines.py \
  --dataset swat --data-dir dataset/SWaT \
  --supervised-mixed-split \
  --methods rf mlp --output-dir results/sota_rf_mlp_swat_fixed \
  --out-log results/logs/rf_mlp_swat_fixed.log

# 3. ToN_IoT: RF + MLP (use supervised-mixed-split on the chrono file)
echo "Running ToN_IoT RF and MLP (Supervised)..."
$PYTHON_EXE baselines/window_baselines.py \
  --dataset ton_iot --data-dir dataset/TON_loT \
  --supervised-mixed-split \
  --mixed-files Processed_datasets/Processed_Linux_dataset/linux_memory1.csv \
  --methods rf mlp --output-dir results/sota_rf_mlp_toniot_fixed \
  --out-log results/logs/rf_mlp_toniot_fixed.log

echo "Missing baselines run completed at $(date)"
