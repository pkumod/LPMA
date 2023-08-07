#include "gpma1.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>

using namespace std;


vector<string> datasets;


bool
test_rebalance_len(thrust::host_vector<KEY_TYPE> &h_keys, \
	SIZE_TYPE node_num, SIZE_TYPE edge_num, SIZE_TYPE group_num) {

	DEV_VEC_SIZE rebalance_stat(32, 0);

	GPMA gpma;
	init_csr_gpma(gpma, node_num, rebalance_stat);

	DEV_VEC_KEY keys;
	DEV_VEC_VALUE values;
	SIZE_TYPE std_group_size = (edge_num + group_num - 1) / group_num;

	for (int i = 0; i < group_num; ++i) {
		SIZE_TYPE bg = i * std_group_size;
		SIZE_TYPE ed = min((i + 1) * std_group_size, edge_num);
		if (bg >= ed)
			break;
		SIZE_TYPE group_size = ed - bg;

		values.resize(group_size, 1);
		keys.resize(group_size);
		cErr(cudaMemcpy(RAW_PTR(keys), RAW_PTR(h_keys) + bg, sizeof(KEY_TYPE) * group_size, cudaMemcpyHostToDevice));

		update_gpma(gpma, keys, values, rebalance_stat);
	}

	KEY_TYPE sum = 0;
	for (int i = 0; i < rebalance_stat.size(); ++i) {
		cout << i << ", " << rebalance_stat[i] << endl;
		sum += (rebalance_stat[i] << i);
	}
	cout << "total rebalance length is: " << sum << endl;

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

	if (test_rebalance_len(h_keys, node_num, edge_num, group_n))
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
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (KEY_TYPE)2048*1024*1024));
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
