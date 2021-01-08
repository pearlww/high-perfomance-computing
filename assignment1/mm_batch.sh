#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=matmult_c.gcc

# define the mkn values in the MKN variable
#
SIZES="50 50 50"

# define the permutation type in PERM
#
PERM="kmn"

# uncomment and set a reasonable BLKSIZE for the blk version
#
# BLKSIZE=1

# enable(1)/disable(0) result checking
export MATMULT_RESULTS=0	  #print result matrices (in Matlab format, def: 0)
export MATMULT_COMPARE=1   # control result comparison (def: 1)
#export MFLOPS_MIN_T = 3     # the minimum run-time (def: 3.0 s)
export MFLOPS_MAX_IT= infinity  # max. no of iterations;


# start the collect command with the above settings
for S in $SIZES
do
    ./$EXECUTABLE $PERM $S $S $S $BLKSIZE
done
