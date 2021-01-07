TARGET	= libmatmult.so
<<<<<<< HEAD
LIBSRCS	= matmult_mkn.c matmult_nmk.c matmult_nkm.c matmult_knm.c matmult_kmn.c matmult_nat.c  matmult_lib.c
=======
LIBSRCS	= matmult_nat.c matmult_lib.c matmult_kmn.c
>>>>>>> b4b6c96db473fa790d2a7fb20304c89e1ab077e7
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
