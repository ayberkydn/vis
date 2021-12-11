#!/bin/bash
#
#SBATCH -p palamut-cuda                     
#SBATCH --exclude palamut9                     
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH -A aaydin                    # Kullanici adi
#SBATCH -J sweep-job                 # Gonderilen isin ismi
#SBATCH -n 1                         # Ayni gorevden kac adet calistirilacak?
#SBATCH -N 1
#SBATCH --cpus-per-task 16           # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --gres=gpu:2                 # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH --time=0-04:01:00             # Sure siniri koyun.


CUDA_VISIBLE_DEVICES=0 singularity run --nv deep-learning_latest.sif wandb agent ayberkydn/vis/zolkri6c &
CUDA_VISIBLE_DEVICES=0 singularity run --nv deep-learning_latest.sif wandb agent ayberkydn/vis/zolkri6c &
CUDA_VISIBLE_DEVICES=1 singularity run --nv deep-learning_latest.sif wandb agent ayberkydn/vis/zolkri6c &
CUDA_VISIBLE_DEVICES=1 singularity run --nv deep-learning_latest.sif wandb agent ayberkydn/vis/zolkri6c &