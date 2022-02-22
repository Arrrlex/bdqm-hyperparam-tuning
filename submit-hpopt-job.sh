#PBS -N train-amptorch
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=2gb
#PBS -l walltime=02:00:00
#PBS -q pace-ice-gpu

cd $PBS_O_WORKDIR
echo "Started on `/bin/hostname`"
echo "Nodes chosen are:"
cat $PBS_NODEFILE

module purge
module load anaconda3/2021.05
conda run -n bdqm-hpopt python hpopt/train.py
