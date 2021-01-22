TARGET_1	= possion_1gpu
TARGET_2	= possion_2gpu

OBJS	= print.o alloc3d.o alloc3d_gpu.o transfer3d_gpu.o jacobi.o 
OBJS_1 = main.o
OBJS_2 = main_2gpu.o

OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_80 -lineinfo


CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

LIBS	= 

CUDA_PATH ?= /appl/cuda/11.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

all: $(TARGET_1) $(TARGET_2) 

$(TARGET_1): $(OBJS) $(OBJS_1) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(OBJS_1)  $(LIBS)

$(TARGET_2): $(OBJS) $(OBJS_2) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(OBJS_2)  $(LIBS)

.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CCC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f core *.o *~

realclean: clean
	@/bin/rm -f $(TARGET)

# dependencies
main.o: main.cu print.h jacobi.h
main_2gpu.o: main_2gpu.cu print.h jacobi.h  
print.o: print.h