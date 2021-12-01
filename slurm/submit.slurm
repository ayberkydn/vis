#!/bin/bash
#
#SBATCH -p palamut-cuda              # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A aaydin                    # Kullanici adi
#SBATCH -J myvissomethingsomething   # Gonderilen isin ismi
#SBATCH -o myoutputtt.out            # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1                 # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                         # Gorev kac node'da calisacak?
#SBATCH -n 1                         # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16           # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=0-16:00:00               # Sure siniri koyun.

# NVIDIA GPU kullanmak icin --nv opsiyonunu kullanin.
singularity run --nv deep-learning_latest.sif python src/main.py
