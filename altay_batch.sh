#!/bin/bash
#SBATCH -J "example"
#SBATCH -o slurm.%j.out                  # cikti dosyasi %j => is no olacak, bu cikti parametrelerini vermek gerekli degildir.
#SBATCH -e slurm.%j.err                  # hata ciktisi dosyasi, bu cikti parametrelerini vermek gerekli degildir.
#SBATCH -p rtx8000
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 0-00:30               # 5 saat demek (D-HH:MM)

#SBATCH --mail-type=END,FAIL                  # is bitince yada sorun cikarsa mail at
#SBATCH --mail-user=aa10947@nyu.edu        # mail atilacak adres

module load anaconda3/2024.02  
source /ari/progs/ANACONDA/Anaconda3-2024.06-1-python-3.12/etc/profile.d/conda.sh
conda activate talkswap

cd /ari/users/mmustu/repos/TalkSwap
python /ari/users/mmustu/repos/TalkSwap/train.py
