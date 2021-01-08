TARGET	= libmatmult.so
LIBSRCS	= matmult_nat.c matmult_lib.c matmult_blk.c
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
