MFLOPS_MAX_IT=0 nv-nsight-cu-cli\
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section ComputeWorkloadAnalysis \
    ./possion_1gpu 100 100 1 0


