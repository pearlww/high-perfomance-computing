module load gcc
make clean
make

time ./poisson_j 200 1000 0.001 20
time ./poisson_gs 200 1000 0.001 20