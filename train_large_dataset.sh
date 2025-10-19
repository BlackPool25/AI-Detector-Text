#!/bin/bash
# Training Scripts for 8.8M Dataset
# Choose the strategy that fits your needs

echo "=================================================="
echo "AI TEXT DETECTION - LARGE DATASET TRAINING"
echo "=================================================="
echo ""
echo "Dataset Size: 8.8M samples"
echo "  - AI: 6,585,718 (74.7%)"
echo "  - Human: 2,233,443 (25.3%)"
echo ""
echo "Available Training Strategies:"
echo ""

# Activate virtual environment
cd /home/lightdesk/Projects/Text
source .venv/bin/activate

read -p "Choose training strategy (1-5): " choice

case $choice in
  1)
    echo ""
    echo "🎯 STRATEGY 1: Quick Test (10K samples, ~30 min)"
    python -m src.training.train_script \
      --data-dir ./processed_data \
      --load-mode all \
      --ai-sample 5000 \
      --human-sample 5000 \
      --num-epochs 2 \
      --batch-size 16 \
      --gradient-accumulation-steps 4 \
      --output-dir ./outputs/test_10k
    ;;
    
  2)
    echo ""
    echo "⚡ STRATEGY 2: Balanced 1M Dataset (500K+500K, ~8-12 hours)"
    nohup python -m src.training.train_script \
      --data-dir ./processed_data \
      --load-mode all \
      --ai-sample 500000 \
      --human-sample 500000 \
      --num-epochs 5 \
      --batch-size 16 \
      --gradient-accumulation-steps 4 \
      --bf16 True \
      --output-dir ./outputs/balanced_1M \
      > training_1M.log 2>&1 &
    echo "Training started in background!"
    echo "Monitor with: tail -f training_1M.log"
    echo "Process ID: $!" | tee training_pid.txt
    ;;
    
  3)
    echo ""
    echo "🔥 STRATEGY 3: Balanced 4.4M Dataset (2.2M+2.2M, ~36-48 hours)"
    echo "This uses ALL human data + equivalent AI data"
    nohup python -m src.training.train_script \
      --data-dir ./processed_data \
      --load-mode all \
      --ai-sample 2233443 \
      --human-sample 2233443 \
      --num-epochs 5 \
      --batch-size 16 \
      --gradient-accumulation-steps 4 \
      --bf16 True \
      --output-dir ./outputs/balanced_4.4M \
      > training_4.4M.log 2>&1 &
    echo "Training started in background!"
    echo "Monitor with: tail -f training_4.4M.log"
    echo "Process ID: $!" | tee training_pid.txt
    ;;
    
  4)
    echo ""
    echo "🚀 STRATEGY 4: Full 8.8M Dataset (48-72 hours)"
    echo "⚠️  WARNING: This is MASSIVE and will take 2-3 DAYS!"
    echo "The class imbalance (75% AI, 25% Human) may affect performance."
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      nohup python -m src.training.train_script \
        --data-dir ./processed_data \
        --load-mode all \
        --num-epochs 3 \
        --batch-size 16 \
        --gradient-accumulation-steps 4 \
        --bf16 True \
        --output-dir ./outputs/full_8.8M \
        > training_8.8M.log 2>&1 &
      echo "Training started in background!"
      echo "Monitor with: tail -f training_8.8M.log"
      echo "Process ID: $!" | tee training_pid.txt
    else
      echo "Cancelled."
    fi
    ;;
    
  5)
    echo ""
    echo "🎨 STRATEGY 5: Custom Configuration"
    echo ""
    read -p "AI samples (or 'all'): " ai_samples
    read -p "Human samples (or 'all'): " human_samples
    read -p "Number of epochs: " epochs
    read -p "Output directory: " output_dir
    
    if [ "$ai_samples" = "all" ]; then
      ai_arg=""
    else
      ai_arg="--ai-sample $ai_samples"
    fi
    
    if [ "$human_samples" = "all" ]; then
      human_arg=""
    else
      human_arg="--human-sample $human_samples"
    fi
    
    nohup python -m src.training.train_script \
      --data-dir ./processed_data \
      --load-mode all \
      $ai_arg \
      $human_arg \
      --num-epochs $epochs \
      --batch-size 16 \
      --gradient-accumulation-steps 4 \
      --bf16 True \
      --output-dir $output_dir \
      > training_custom.log 2>&1 &
    echo "Training started in background!"
    echo "Monitor with: tail -f training_custom.log"
    echo "Process ID: $!" | tee training_pid.txt
    ;;
    
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo ""
echo "=================================================="
echo "Training Configuration Complete"
echo "=================================================="
