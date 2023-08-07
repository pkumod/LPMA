#include "gpma.cuh"
#include "util.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>

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

	GPMA gpma;
	init_csr_gpma(gpma, node_num);

	DEV_VEC_KEY keys;
	DEV_VEC_VALUE values;
	SIZE_TYPE std_group_size = (edge_num + group_num - 1) / group_num;
	int cnt = 0;
	int i;

	for (i = 0; i < group_num; ++i) {
		SIZE_TYPE bg = i * std_group_size;
		SIZE_TYPE ed = min((i + 1) * std_group_size, edge_num);
		if (cnt)
			break;
		if (ed == edge_num)
			++cnt;
		SIZE_TYPE group_size = ed - bg;

		values.resize(group_size, 1);
		keys.resize(group_size);
		cErr(cudaMemcpy(RAW_PTR(keys), RAW_PTR(h_keys) + bg, sizeof(KEY_TYPE) * group_size, cudaMemcpyHostToDevice));

TimeKeeper t1;
cErr(cudaDeviceSynchronize());

	update_gpma(gpma, keys, values);

cErr(cudaDeviceSynchronize());

t += t1.checkTime("");

if(i%(group_num/5)==0){
cout << i<<": "<<t/(i+1)<< endl;
}

t += t1.checkTime("");

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
	cout << "node num is:" << node_num << ", edge num is:" << edge_num << endl;
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
	group_n=edge_num/group_n;
	cout << "node num is:" << node_num << ", edge num is:" << edge_num <<", group num is:" << group_n << endl;

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
	datasets.push_back(argv[3]);
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num, group_num);
	}
	return 0;
}
