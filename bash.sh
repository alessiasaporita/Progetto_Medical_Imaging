source /homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-*/etc/profile.d/conda.sh
conda activate nnunet
cd /work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging

export nnUNet_raw="/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/nnUNet_raw"
export nnUNet_preprocessed="/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/nnUNet_preprocessed"
export nnUNet_results="/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging/nnUNet_results"


srun -Q --immediate=10 --mem=20G --partition=all_serial --gres=gpu:1 --nodes=1 --time 4:00:00  --pty --account=grana_neuro -w ailb-login-03 bash