TARGET	= libmatmult.so
LIBSRCS	= matmult_nat.c matmult_lib.c matmult_blk.c 
			matmult_mnk.c matmult_mkn.c matmult_nmk.c
			matmult_nkm.c matmult_kmn.c matmult_knm.c
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g -O3
PIC	= -fPIC

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)


clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
