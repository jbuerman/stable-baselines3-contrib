#!/bin/bash
set -euo pipefail

SIF="<path to sif>.sif"
BIND=""  # "--bind <any bind>"

# Jobs to run
games=("BattleZone"  "NameThisGame" "Phoenix")
repeats=(0)

# Partitions to target (edit as needed)
partitions=("<partition>")

pick_resources() {
  case "$1" in
    *)                          echo "24 60:00:00"   ;; # default
  esac
}

for part in "${partitions[@]}"; do
  read -r CPUS WALLTIME <<<"$(pick_resources "$part")"

  for game in "${games[@]}"; do
    for rep in "${repeats[@]}"; do

      job_name="${game:0:3}${rep}"

      sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --job-name=${job_name}
#SBATCH --time=${WALLTIME}
#SBATCH -p ${part}
#SBATCH --account=<account>
set -euo pipefail

module load apptainer

apptainer run --env WANDB_MODE=offline --nv $BIND --pwd <working dir> "$SIF" --game ${game} --repeat ${rep}

module unload apptainer
EOF

    done
  done
done
