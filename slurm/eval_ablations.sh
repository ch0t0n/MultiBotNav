#!/bin/bash
# ============================================================
# eval_ablations.sh — SAFE version (per-job CSVs, no conflicts)
#
# Covers experiments that require separate evaluation passes:
#   ablation_reward       — 50-ep eval for terminal-condition tracking
#   ablation_uncertainty  — cross-evaluation matrix (train × eval noise)
#   dr                    — in-distribution and OOD wind evaluation
#
# Note: ablation_obs results are still read directly from training
# evaluations.npz files by analyze_results.py; no separate eval
# pass is needed for that experiment.
#
# Usage (arg1=experiment, arg2=robot_type, default arg2=uav):
#   sbatch --array=0-89  eval_ablations.sh ablation_reward      uav
#   sbatch --array=0-89  eval_ablations.sh ablation_reward      wheeled
#   sbatch --array=0-479 eval_ablations.sh ablation_uncertainty uav
#   sbatch --array=0-479 eval_ablations.sh ablation_uncertainty wheeled
#   sbatch --array=0-359 eval_ablations.sh dr                   uav
#   sbatch --array=0-359 eval_ablations.sh dr                   wheeled
#
# Grid sizes:
#   ablation_reward       uav  : 3 cond × 10 sets × 3 seeds           =  90 jobs
#   ablation_reward       wheel: 3 cond × 10 sets × 3 seeds           =  90 jobs
#   ablation_uncertainty  uav  : 4 train × 4 eval × 10 sets × 3 seeds = 480 jobs
#   ablation_uncertainty  wheel: 4 train × 4 eval × 10 sets × 3 seeds = 480 jobs
#   dr                    uav  : 3 dr × 10 sets × 4 robots × 3 seeds  = 360 jobs
#   dr                    wheel: 3 dr × 10 sets × 4 robots × 3 seeds  = 360 jobs
# ============================================================

#SBATCH --job-name=eval_ablation
#SBATCH --output=logs/slurm_outputs/eval_ablations/%x_%j.out
#SBATCH --error=logs/slurm_errors/eval_ablations/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --export=NONE

EXPERIMENT=${1:-ablation_uncertainty}
ROBOT_TYPE=${2:-uav}

# Per-experiment output root (separate per robot type)
OUT_ROOT="logs/results/tmp/${EXPERIMENT}_${ROBOT_TYPE}"
mkdir -p "$OUT_ROOT"

seeds=(42 123 9999)
num_seeds=${#seeds[@]}
index=$SLURM_ARRAY_TASK_ID

# Robot-type-specific env set list
if [ "$ROBOT_TYPE" == "uav" ]; then
    sets=(1 2 3 4 5 6 7 8 9 10)
else
    sets=(1 2 3 4 5 6 7 8 9 10)
fi
num_sets=${#sets[@]}

if [ "$EXPERIMENT" == "ablation_reward" ]; then
    conditions=("full" "no_term" "no_path")
    num_conditions=${#conditions[@]}

    seed_idx=$(( index % num_seeds ))
    set_idx=$(( (index / num_seeds) % num_sets ))
    cond_idx=$(( index / (num_seeds * num_sets) ))

    condition=${conditions[$cond_idx]}
    set=${sets[$set_idx]}
    seed=${seeds[$seed_idx]}

    OUT_DIR="${OUT_ROOT}/${condition}/set${set}"
    mkdir -p "$OUT_DIR"
    OUT_CSV="${OUT_DIR}/result_${index}.csv"

    echo "eval | ablation_reward | robot=$ROBOT_TYPE | condition=$condition | set=$set | seed=$seed"
    echo "Output: $OUT_CSV"

    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --robot_type $ROBOT_TYPE \
        --set $set --num_robots 3 --seed $seed \
        --experiment ablation_reward --ablation $condition \
        --output_csv $OUT_CSV --n_eval_eps 50

# ============================================================
# ── Ablation: uncertainty (cross-eval matrix, all env sets) ──
# ============================================================
elif [ "$EXPERIMENT" == "ablation_uncertainty" ]; then
    train_modes=("full" "wind_only" "act_only" "deterministic")
    eval_modes=("full" "wind_only" "act_only" "deterministic")

    seed_idx=$(( index % num_seeds ))
    set_idx=$(( (index / num_seeds) % num_sets ))
    train_idx=$(( (index / (num_seeds * num_sets)) % 4 ))
    eval_idx=$(( index / (num_seeds * num_sets * 4) ))

    train_mode=${train_modes[$train_idx]}
    eval_mode=${eval_modes[$eval_idx]}
    seed=${seeds[$seed_idx]}
    set=${sets[$set_idx]}

    OUT_DIR="${OUT_ROOT}/train_${train_mode}_eval_${eval_mode}/set${set}"
    mkdir -p "$OUT_DIR"
    OUT_CSV="${OUT_DIR}/result_${index}.csv"

    echo "eval | ablation_uncertainty | robot=$ROBOT_TYPE | train=$train_mode | eval=$eval_mode | set=$set | seed=$seed"
    echo "Output: $OUT_CSV"

    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --robot_type $ROBOT_TYPE \
        --set $set --num_robots 3 --seed $seed \
        --experiment ablation_uncertainty --ablation $train_mode \
        --eval_uncertainty_mode $eval_mode \
        --output_csv $OUT_CSV --n_eval_eps 50

# ============================================================
# ── Domain randomization ─────────────────────────────────────
# ============================================================
elif [ "$EXPERIMENT" == "dr" ]; then

    dr_modes=("none" "wind" "full")
    # Both robot types now test 2-5 robots
    robots=(2 3 4 5)
    num_robots=${#robots[@]}

    seed_idx=$(( index % num_seeds ))
    robot_idx=$(( (index / num_seeds) % num_robots ))
    set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
    dr_idx=$(( index / (num_seeds * num_robots * num_sets) ))

    dr_mode=${dr_modes[$dr_idx]}
    set=${sets[$set_idx]}
    num_robots_value=${robots[$robot_idx]}
    seed=${seeds[$seed_idx]}

    OUT_DIR="${OUT_ROOT}/${dr_mode}/set${set}_N${num_robots_value}"
    mkdir -p "$OUT_DIR"

    echo "eval | dr | robot=$ROBOT_TYPE | mode=$dr_mode | set=$set | robots=$num_robots_value | seed=$seed"

    # In-distribution
    OUT_CSV_IN="${OUT_DIR}/inDist_${index}.csv"
    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --robot_type $ROBOT_TYPE \
        --set $set --num_robots $num_robots_value --seed $seed \
        --experiment dr --ablation $dr_mode \
        --eval_wind_min 0.0 --eval_wind_max 0.5 \
        --output_csv $OUT_CSV_IN --n_eval_eps 50

    # OOD
    OUT_CSV_OOD="${OUT_DIR}/OOD_${index}.csv"
    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --robot_type $ROBOT_TYPE \
        --set $set --num_robots $num_robots_value --seed $seed \
        --experiment dr --ablation $dr_mode \
        --eval_wind_min 0.5 --eval_wind_max 2.0 \
        --output_csv $OUT_CSV_OOD --n_eval_eps 50

else
    echo "ERROR: unknown experiment '${EXPERIMENT}'."
    echo "Usage:"
    echo "  sbatch --array=0-89  eval_ablations.sh ablation_reward      uav"
    echo "  sbatch --array=0-479 eval_ablations.sh ablation_uncertainty uav"
    echo "  sbatch --array=0-359 eval_ablations.sh dr                   uav"
    exit 1
fi

wait
