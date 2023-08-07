#include "rpma.cuh"
#include "util.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>
#include "cusparse.h"

using namespace std;

__global__
void key_copy_kernel(KEY_TYPE *keys, SIZE_TYPE* rowidx, SIZE_TYPE *colidx, SIZE_TYPE size) {
	SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        rowidx[i] = (keys[i] >> 32);
        colidx[i] = keys[i];
    }
}



vector<string> datasets;


bool
test_update(thrust::host_vector<KEY_TYPE> &h_keys, SIZE_TYPE node_num, SIZE_TYPE edge_num, SIZE_TYPE group_num) {

KEY_TYPE t = 0;

	SIZE_TYPE std_group_size = (edge_num + group_num - 1) / group_num;
	int cnt = 0;
	int i;
	for (i = 0; i < group_num; ++i) {
		SIZE_TYPE ed = min((i + 1) * std_group_size, edge_num);
		if (ed == edge_num)
			++cnt;
		if (cnt)
			break;

		DEV_VEC_SIZE coo_rowidx(ed);
		DEV_VEC_SIZE coo_colidx(ed, 0);
		DEV_VEC_KEY d_keys(ed);
		cErr(cudaDeviceSynchronize());
		cErr(cudaMemcpy(RAW_PTR(d_keys), RAW_PTR(h_keys), sizeof(KEY_TYPE) * ed, cudaMemcpyHostToDevice));

TimeKeeper t1;
cErr(cudaDeviceSynchronize());
thrust::sort(d_keys.begin(), d_keys.end());
cErr(cudaDeviceSynchronize());
t += t1.checkTime("");

	key_copy_kernel<<<256, 256>>>(RAW_PTR(d_keys), RAW_PTR(coo_rowidx), RAW_PTR(coo_colidx), ed);

	DEV_VEC_SIZE csr_rowoffset(node_num + 1);

	cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) assert(0);

    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) assert(0);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

TimeKeeper t2;
cErr(cudaDeviceSynchronize());
    status= cusparseXcoo2csr(handle, (int *)RAW_PTR(coo_rowidx), ed, node_num,
                             (int *)RAW_PTR(csr_rowoffset), CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) assert(0);
cErr(cudaDeviceSynchronize());
t += t2.checkTime("");

	}
cout << t/i << endl;
	return true;
}

void
read_edgelist(string &filename, thrust::host_vector<KEY_TYPE> &h_keys, SIZE_TYPE &node_num, SIZE_TYPE &edge_num, SIZE_TYPE edge_n) {
	ifstream e_file;
	e_file.open((filename + ".edgelist").c_str());

	e_file >> node_num;
	e_file >> edge_num;
	if (edge_n)
		edge_num = edge_n;

	for (int i = 0; i < edge_num; ++i) {
		KEY_TYPE src, dst;
		e_file >> src;
		e_file >> dst;
		assert(src < node_num && dst < node_num);
		h_keys.push_back(src<<32 | dst);
	}
	e_file.close();
}


void
test_all(string dataset, SIZE_TYPE edge_n, SIZE_TYPE group_n) {
	cout << "reading edgelist and timestamps" << endl;
	SIZE_TYPE node_num, edge_num;
	thrust::host_vector<KEY_TYPE> h_keys;
	read_edgelist(dataset, h_keys, node_num, edge_num, edge_n);
	cout << "node num is:" << node_num << ", edge num is:" << edge_num << endl;

	if (test_update(h_keys, node_num, edge_num, group_n))
		cout << "query and update with timestamp test success" << endl;
	else {
		cout << "query and update with timestamp test fail" << endl;
		return;
	}
}


int
main(int args, char *argv[]) {

	SIZE_TYPE edge_num = atoi(argv[1]);
	SIZE_TYPE group_num = atoi(argv[2]);

	cErr(cudaDeviceSynchronize());
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048*1024*1024));
	cErr(cudaDeviceSynchronize());

	//datasets.push_back("pokec");
	datasets.push_back("livejournal");
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num, group_num);
	}
	return 0;
}
