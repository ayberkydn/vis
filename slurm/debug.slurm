#!/bin/bash
#
#SBATCH -p debug                     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A aaydin                    # Kullanici adi
#SBATCH -J myvissomethingsomething   # Gonderilen isin ismi
#SBATCH -o myoutputtt.out            # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1                 # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 4                         # Gorev kac node'da calisacak?
#SBATCH -n 4                         # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16           # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=0:01:00               # Sure siniri koyun.

# NVIDIA GPU kullanmak icin --nv opsiyonunu kullanin.
singularity run --nv deep-learning_latest.sif wandb agent ayberkydn/some-vis-thingsweep/sbilkkd2
#singularity run --nv deep-learning_latest.sif python src/main.py
