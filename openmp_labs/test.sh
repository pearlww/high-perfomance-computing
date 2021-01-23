module load studio

#cc -g -fast -xautopar -xloopinfo -c mxv.c
#cc -g -fast -xautopar -xloopinfo -xreduction -c intergration.c

cc -g -fast -xopenmp -xloopinfo -c mxv.c
cc -g -fast -xopenmp -xloopinfo -c intergration.c

time OMP_NUM_THREADS=4 a.out 
# clock measure the CPU time used by process, not the wall-clock time.

