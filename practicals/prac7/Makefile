
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

trid:	trid.cu trid_gold.cpp Makefile
	nvcc trid.cu trid_gold.cpp -o trid \
	$(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm trid
