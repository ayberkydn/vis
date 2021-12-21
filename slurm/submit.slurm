#!/bin/bash
#
#SBATCH -p palamut-cuda                     
#SBATCH --exclude palamut9                     
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH -A norom1                    # Kullanici adi
#SBATCH -J submit-job                 # Gonderilen isin ismi
#SBATCH -n 1                         # Ayni gorevden kac adet calistirilacak?
#SBATCH -N 1
#SBATCH --cpus-per-task 16           # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --gres=gpu:1                 # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH --time=0-1:00:00             # Sure siniri koyun.

CUDA_VISIBLE_DEVICES=0 singularity run --nv vis_latest.sif python src/main.py &

wait
