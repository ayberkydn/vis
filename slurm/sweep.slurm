#!/bin/bash
#
#SBATCH -p palamut-cuda                     
#SBATCH --exclude palamut9                     
#SBATCH -A aaydin                    # Kullanici adi
#SBATCH -J sweep-job                 # Gonderilen isin ismi
#SBATCH -n 1                         # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16           # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --gres=gpu:1                 # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH --time=0-10:00:00             # Sure siniri koyun.


singularity run --nv deep-learning_latest.sif wandb agent ayberkydn/lr-deneme/vm3mvicd
#SBATCH -N 1                         # Gorev kac node'da calisacak?
#SBATCH --gpus-per-task 1