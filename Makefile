NVCC = nvcc -arch=sm_60 -lcudadevrt -rdc=true -O3 #-g -G# --ptxas-options=-v
C_FLAGS = -std=c++11
DEBUG_FLAGS = -g -G


ALGO_RPMA: test_algos.cu util.cuh
	$(NVCC) -o ALGO_RPMA test_algos.cu $(CUDA_FLAGS) $(C_FLAGS) -lnvgraph
	
RPMA: test_update_correctness_rpma.cu rpma.cuh util.cuh
	$(NVCC) -o RPMA test_update_correctness_rpma.cu $(CUDA_FLAGS) $(C_FLAGS)

GPMA: test_gpma.cu gpma.cuh util.cuh
	$(NVCC) -o GPMA test_gpma.cu $(CUDA_FLAGS) $(C_FLAGS)

CSR_RPMA: test_csr.cu rpma.cuh util.cuh
	$(NVCC) -o CSR_RPMA test_csr.cu $(CUDA_FLAGS) $(C_FLAGS)

CUSPARSE: test_cusparse.cu rpma.cuh util.cuh
	$(NVCC) -o CUSPARSE test_cusparse.cu $(CUDA_FLAGS) $(C_FLAGS) -lcusparse

UPDATE_LPMA: test_update_rpma.cu rpma.cuh util.cuh
	$(NVCC) -o UPDATE_LPMA test_update_rpma.cu $(CUDA_FLAGS) $(C_FLAGS)

UPDATE_GPMA: test_update_gpma.cu gpma.cuh util.cuh
	$(NVCC) -o UPDATE_GPMA test_update_gpma.cu $(CUDA_FLAGS) $(C_FLAGS)

REB_LEN_GPMA: test_rebalance_len_gpma.cu gpma.cuh util.cuh
	$(NVCC) -o REB_LEN_GPMA test_rebalance_len_gpma.cu $(CUDA_FLAGS) $(C_FLAGS)

REB_LEN_RPMA: test_rebalance_len_rpma.cu rpma.cuh util.cuh
	$(NVCC) -o REB_LEN_RPMA test_rebalance_len_rpma.cu $(CUDA_FLAGS) $(C_FLAGS)
