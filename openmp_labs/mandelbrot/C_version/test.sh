# echo "#threds wall user sys";
# for t in 1 2 4 8;
#     do echo -n "$t ";
#     OMP_NUM_THREADS=$t time -f "%e %U %S" ./mandelbrot 5000;
# done

#OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 collect ./mandelbrot
#analyzer test.1.er

# OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 OMP_SCHEDULE=dynamic,50 collect ./mandelbrot
# analyzer test.2.er

echo "#threds wall user sys";
for t in 1 2 4 8;
    do echo -n "$t ";
    OMP_NUM_THREADS=$t OMP_SCHEDULE=dynamic,50 time -f "%e %U %S" ./mandelbrot 5000;
done