
INC	:= -I$(CUDA_HOME)/include -I. -I../headers
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all: 		prac0

prac0:	kernel.cu host.cu Makefile
		nvcc kernel.cu host.cu -o prac0 \
		$(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f prac0
