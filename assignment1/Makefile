TARGET	= libmatmult.so
LIBSRCS	= matmult_mkn.c matmult_nmk.c matmult_nkm.c matmult_knm.c matmult_kmn.c matmult_nat.c  matmult_lib.c
LIBOBJS	= $(LIBSRCS:.c=.o)

OPT	= -g 
PIC	= -fPIC

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)


clean:
	@/bin/rm -f core core.* $(LIBOBJS) 
