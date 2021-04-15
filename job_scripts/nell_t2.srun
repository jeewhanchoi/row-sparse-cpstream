#!/bin/bash
#SBATCH --partition=fat         ### Partition (like a queue in PBS)
#SBATCH --job-name=nell_t2       ### Job Name
#SBATCH --output=nell_t2.out     ### File in which to store job output
#SBATCH --error=nell_t2.err      ### File in which to store job error messages
#SBATCH --time=0-12:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Number of nodes needed for the job
#SBATCH --mem=512G              ### Memory per CPU
#SBATCH --ntasks-per-node=1     ### Number of tasks to be launched per Node
#SBATCH --cpus-per-task=56      ### Number of threads per task (OMP threads)
#SBATCH --account=hpctensor     ### Anellount used for job submission

export OMP_NUM_THREADS=56
export KMP_AFFINITY=granularity=fine,compact,1

./build/Linux-x86_64/bin/splatt cpd -v --stream=1 -r 32 -t 2 --reg=frob,1e-2,1 ../hpctensor/nell-1M.tns
