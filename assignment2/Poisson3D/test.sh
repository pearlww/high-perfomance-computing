module load gcc
make clean
make

# SIZES="100 200 400"

# for S in $SIZES
# do
#     time ./poisson_j $S 10000 1 0 >> $S.dat
# done


# PERM="jacobi"
# LOGEXT=$PERM.dat
# /bin/rm -f $LOGEXT
# time ./poisson_j 200 1000 1 0 #4 >> $LOGEXT

# PERM="gauss"
# LOGEXT=$PERM.dat
# /bin/rm -f $LOGEXT
time ./poisson_gs 200 1000 1 0 #4 >> $LOGEXT