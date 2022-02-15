#PBS -N train-amptorch-model
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=2gb
#PBS -l walltime=00:20:00
#PBS -q pace-ice-gpu
#PBS -k oe
#PBS -m abe
#PBS -M amckenzie9@gatech.edu

cd $PBS_O_WORKDIR
echo "Started on `/bin/hostname`"
echo "Nodes chosen are:"
cat $PBS_NODEFILE

module load anaconda3/2021.05
# conda run -n bdqm-hpopt pip install git+https://github.com/medford-group/amptorch.git@BDQM_VIP_2022Feb
# conda run -n bdqm-hpopt pip install git+https://github.com/ulissigroup/amptorch.git@MCSH_paper1_lmdb
conda run -n bdqm-hpopt python scripts/test_new_branch_nicole.py
