#!/bin/bash
#SBATCH --job-name=portfoliotest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --time=20:00:00
#SBATCH -o /scratch/gpfs/iywang/learn_robust/lropt_results/portfolio/portfolio_test_%A_.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=iabirina@hotmail.com

module purge
module load anaconda3/2023.9
conda activate lropt

python portfolio/portfolio2.py --foldername /scratch/gpfs/iywang/learn_robust/lropt_results/portfolio/old35/results7/ --eta 0.30

# python portfolio/plot_avg.py --foldername /scratch/gpfs/iywang/learn_robust/lropt_results/portfolio/old33/

# python portfolio/MIP/plots.py --foldername /scratch/gpfs/iywang/mro_results/portfolio/new/m30_K1000_r10/

# (0ld2) 10,20,30,15,5,20(+0.3I), 20 (with MV start, + 0.3I)
# (seed = 1) 10, 5, 15 (mu=0.02), 20, 10, lam=0.1, mumult=1.001
# seed = 1, mu = 0.02 m=10 r+40, m=5, 8, 5 r+50,10,8


# 0 eta = 0.1, m=8, r+50
# m=5
# m=20

# 0 eta=0.01, m=20, r+50, eta=0.05, eta=0.15, eta=0.2, eta=0.3, eta=0.4, eta=0.5

# 0 eta=0.01, m=5, r, eta=0.05, eta=0.1, eta=0.15, eta=0.2, eta=0.3, eta=0.4, eta=0.5

#0 eta=0.05 m=5, r, m=8, 20
#3 eta=0.4, m=5, m=8,20

#0 eta=0.01, m=5, eta=0.05, eta=0.1, eta=0.2, eta=0.3, eta=0.4, seed=15

# r+10

# r 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2

#old13 n=5 r 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3 seed=25

#old14 n=5 r 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3 seed=25 iters=3000

#old15 n=10 r 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3 seed=25 iters=3000

#old16 n=10 r 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3 seed=27 iters=3000

#old17 n=5 r  0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3 seed=27 iters=3000

#old18 n=5 r 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.13 0.15 0.18 0.20 0.25 0.30

#old19 n=10
#old20 n=20 lr=0.01
#old21 n=20 0.01 0.03 0.05 0.10 (no random)  0.01 0.03 0.05 0.08 0.1 0.15 0.20(lr=0.0001)
#old22 n=10 r=r+30 all 17 (to comb with old19)
#old23 n=5 r=r+30 all 17 (to comb with old18)
#old24 iters 3000, epslst 300->100, newdata 20000->3000, batch 10
#old25 n=5 batch 30

#old26 n=5 batch 40 all 17 all y 3000 test
#old27 n=5 batch40 8000 test
#old28 n=10, r10

#old29 n=10 batch30, y norm 0.05, 8000test, n=5
#old30 n=5 1y

#old31 n=5 5y 20000test 
#old32 n=10 5y

#old33 n=10 R20 10y 20000test 5par with x
#old34 n=20 R10 10y 20000test 5par with x