# Source 
# CULane (train: 88,000) to Curvelanes 
#idx: 0 (0), 20,000 (1), 40,000 (2), 60,000 (3), 80,000 (4) 
# 20,000 src -> 10, 20, 50, 100
#!/bin/bash

echo "CULane → Curvelanes | tgt_samples=10 | block_idx=0"
python shift_experiment.py --source CULane --target Curvelanes --src_split train --tgt_split valid --src_samples 50000 --tgt_samples 10 --num_runs 100 --block_idx 0 --batch_size 128 --num_calib 100

echo "CULane → Curvelanes | tgt_samples=10 | block_idx=1"
python shift_experiment.py --source CULane --target Curvelanes --src_split train --tgt_split valid --src_samples 50000 --tgt_samples 10 --num_runs 100 --block_idx 1 --batch_size 128 --num_calib 100


echo "Curvelanes → CULane | tgt_samples=10 | block_idx=0"
python shift_experiment.py --source Curvelanes --target CULane --src_split train --tgt_split test --src_samples 50000 --tgt_samples 10 --num_runs 100 --block_idx 0 --batch_size 128 --num_calib 100

echo "Curvelanes → CULane | tgt_samples=10 | block_idx=1"
python shift_experiment.py --source Curvelanes --target Curvelanes --src_split train --tgt_split test --src_samples 50000 --tgt_samples 10 --num_runs 100 --block_idx 1 --batch_size 128 --num_calib 100


# Curvelanes to CULane

# CULane to Shifted CULane
# Gaussian noise [1,10,15,20,25,30,35,40,45,50]
# 

# ps: we want to store mmd values of each run. the tau values for later preprocessing