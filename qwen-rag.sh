#!/bin/bash
#SBATCH --job-name=qwen_rag_chatbot   # Job name
#SBATCH --nodes=1              # Run all processes on a single node
#SBATCH --ntasks=1             # Run a single task
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --mem=40gb             # Memory (adjust as needed)
#SBATCH --cpus-per-task=8      # CPU cores
#SBATCH --partition=gpu_l40 # GPU partition name
#SBATCH --output=qwen_job_%j.log  # Output log file

# --- 1️⃣ Load Anaconda ---
module load anaconda3

# --- 2️⃣ Activate your conda env ---
source /home/du1/21CS30035/anaconda3/bin/activate qwen_env

# --- 4️⃣ Run your script ---
python non_interactive_rag_chatbot.py