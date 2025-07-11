#!/bin/bash
set -e  # Exit on any error

# Stage 1: Batch processing for multiple datasets
# This script processes all datasets under data/data_WACV_spinnerf/ for Stage 1 (depth map generation)

# Configuration
DATA_ROOT="data/data_WACV_spinnerf"
FACTOR=2
STAGE1_ITERS=4001
LOG_DIR="logs/stage1_batch"

# Create log directory
mkdir -p "${LOG_DIR}"

# Main log file
MAIN_LOG="${LOG_DIR}/batch_processing_$(date '+%Y%m%d_%H%M%S').log"

echo "=== SPIn-NeRF Stage 1 Batch Processing ===" | tee -a "${MAIN_LOG}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MAIN_LOG}"
echo "Data root: ${DATA_ROOT}" | tee -a "${MAIN_LOG}"
echo "Factor: ${FACTOR}" | tee -a "${MAIN_LOG}"
echo "Iterations: ${STAGE1_ITERS}" | tee -a "${MAIN_LOG}"
echo "=========================================" | tee -a "${MAIN_LOG}"

# Check if data directory exists
if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: Data directory ${DATA_ROOT} not found!" | tee -a "${MAIN_LOG}"
    exit 1
fi

# Find all dataset directories
datasets=($(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort))

if [ ${#datasets[@]} -eq 0 ]; then
    echo "ERROR: No datasets found in ${DATA_ROOT}" | tee -a "${MAIN_LOG}"
    exit 1
fi

echo "Found ${#datasets[@]} datasets to process:" | tee -a "${MAIN_LOG}"
for dataset in "${datasets[@]}"; do
    echo "  - $(basename "$dataset")" | tee -a "${MAIN_LOG}"
done
echo "" | tee -a "${MAIN_LOG}"

# Process each dataset
success_count=0
failed_datasets=()

for dataset_path in "${datasets[@]}"; do
    dataset_name=$(basename "$dataset_path")
    dataset_log="${LOG_DIR}/${dataset_name}_stage1.log"
    
    echo ">>> Processing dataset: ${dataset_name}" | tee -a "${MAIN_LOG}"
    echo ">>> Dataset path: ${dataset_path}" | tee -a "${MAIN_LOG}"
    echo ">>> Log file: ${dataset_log}" | tee -a "${MAIN_LOG}"
    
    # Validate dataset structure
    echo "Validating dataset structure..." | tee -a "${MAIN_LOG}"
    
    required_items=(
        "${dataset_path}/images"
        "${dataset_path}/images_2" 
        "${dataset_path}/label"
        "${dataset_path}/sparse"
        "${dataset_path}/poses_bounds.npy"
    )
    
    validation_failed=false
    for item in "${required_items[@]}"; do
        if [ ! -e "$item" ]; then
            echo "ERROR: Missing ${item}" | tee -a "${MAIN_LOG}"
            validation_failed=true
        fi
    done
    
    if [ "$validation_failed" = true ]; then
        echo "ERROR: Dataset ${dataset_name} failed validation. Skipping..." | tee -a "${MAIN_LOG}"
        failed_datasets+=("$dataset_name")
        echo "" | tee -a "${MAIN_LOG}"
        continue
    fi
    
    # Move label folder into images_2/ if needed
    echo "Organizing label folder..." | tee -a "${MAIN_LOG}"
    
    if [ -d "${dataset_path}/label" ] && [ ! -d "${dataset_path}/images_2/label" ]; then
        echo "Moving label folder from dataset root to images_2/" | tee -a "${MAIN_LOG}"
        mv "${dataset_path}/label" "${dataset_path}/images_2/label"
        echo "Label folder moved successfully" | tee -a "${MAIN_LOG}"
    elif [ -d "${dataset_path}/images_2/label" ]; then
        echo "Label folder already exists in images_2/, skipping move" | tee -a "${MAIN_LOG}"
    else
        echo "WARNING: No label folder found in dataset root" | tee -a "${MAIN_LOG}"
    fi
    
    # Convert JPG to PNG if needed
    echo "Converting JPG images to PNG format..." | tee -a "${MAIN_LOG}"
    
    # Convert images in images_2/ folder
    if [ -d "${dataset_path}/images_2" ]; then
        jpg_count=$(find "${dataset_path}/images_2" -maxdepth 1 -name "*.jpg" -o -name "*.JPG" | wc -l)
        if [ $jpg_count -gt 0 ]; then
            echo "Found ${jpg_count} JPG images in images_2/, converting to PNG..." | tee -a "${MAIN_LOG}"
            python jpg2png.py "${dataset_path}/images_2" 2>&1 | tee -a "${dataset_log}"
        else
            echo "No JPG images found in images_2/, skipping conversion" | tee -a "${MAIN_LOG}"
        fi
    fi
    
    # Convert images in images_2/label/ folder  
    if [ -d "${dataset_path}/images_2/label" ]; then
        label_jpg_count=$(find "${dataset_path}/images_2/label" -name "*.jpg" -o -name "*.JPG" | wc -l)
        if [ $label_jpg_count -gt 0 ]; then
            echo "Found ${label_jpg_count} JPG label images, converting to PNG..." | tee -a "${MAIN_LOG}"
            python jpg2png.py "${dataset_path}/images_2/label" 2>&1 | tee -a "${dataset_log}"
        else
            echo "No JPG label images found, skipping label conversion" | tee -a "${MAIN_LOG}"
        fi
    fi
    
    # Count images to verify train/test split (now looking for PNG after conversion)
    train_count=$(find "${dataset_path}/images_2" -name "IMG_*.png" -o -name "IMG_*.PNG" | wc -l)
    test_count=$(find "${dataset_path}/images_2" -name "test_IMG_*.png" -o -name "test_IMG_*.PNG" | wc -l)
    total_count=$((train_count + test_count))
    
    echo "Images found after conversion: ${total_count} total (${train_count} train, ${test_count} test)" | tee -a "${MAIN_LOG}"
    
    if [ $total_count -eq 0 ]; then
        echo "ERROR: No images found in ${dataset_path}/images_2 after conversion" | tee -a "${MAIN_LOG}"
        failed_datasets+=("$dataset_name")
        echo "" | tee -a "${MAIN_LOG}"
        continue
    fi
    
    # Prepare LaMa directories
    echo "Preparing LaMa directories..." | tee -a "${MAIN_LOG}"
    rm -rf lama/LaMa_test_images/*
    rm -rf lama/output/label/*
    
    # Record start time for this dataset
    dataset_start_time=$(date +%s)
    
    echo "Starting Stage 1 training for ${dataset_name}..." | tee -a "${MAIN_LOG}"
    
    # Run Stage 1 training
    if python DS_NeRF/run_nerf.py \
        --config DS_NeRF/configs/config.txt \
        --render_factor 1 \
        --prepare \
        --i_weight 1000000000 \
        --i_video 1000000000 \
        --i_feat 4000 \
        --N_iters ${STAGE1_ITERS} \
        --expname "${dataset_name}_stage1" \
        --datadir "${dataset_path}" \
        --factor ${FACTOR} \
        --N_gt 0 \
        2>&1 | tee "${dataset_log}"; then
        
        # Calculate duration
        dataset_end_time=$(date +%s)
        duration=$((dataset_end_time - dataset_start_time))
        
        echo "SUCCESS: ${dataset_name} completed in ${duration} seconds" | tee -a "${MAIN_LOG}"
        success_count=$((success_count + 1))
        
        # Verify output
        if [ -d "lama/LaMa_test_images" ] && [ "$(ls -A lama/LaMa_test_images)" ]; then
            image_count=$(find lama/LaMa_test_images -name "*.png" | wc -l)
            echo "Generated ${image_count} depth images for LaMa processing" | tee -a "${MAIN_LOG}"
        else
            echo "WARNING: No depth images generated in LaMa_test_images" | tee -a "${MAIN_LOG}"
        fi
        
    else
        echo "ERROR: ${dataset_name} failed during training" | tee -a "${MAIN_LOG}"
        failed_datasets+=("$dataset_name")
    fi
    
    echo "----------------------------------------" | tee -a "${MAIN_LOG}"
    echo "" | tee -a "${MAIN_LOG}"
done

# Final summary
echo "=== BATCH PROCESSING COMPLETE ===" | tee -a "${MAIN_LOG}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MAIN_LOG}"
echo "Successfully processed: ${success_count}/${#datasets[@]} datasets" | tee -a "${MAIN_LOG}"

if [ ${#failed_datasets[@]} -gt 0 ]; then
    echo "Failed datasets:" | tee -a "${MAIN_LOG}"
    for failed in "${failed_datasets[@]}"; do
        echo "  - ${failed}" | tee -a "${MAIN_LOG}"
    done
else
    echo "All datasets processed successfully!" | tee -a "${MAIN_LOG}"
fi

echo "Main log: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "Individual logs: ${LOG_DIR}/" | tee -a "${MAIN_LOG}"
echo "===============================" | tee -a "${MAIN_LOG}"

# Exit with error code if any datasets failed
if [ ${#failed_datasets[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi