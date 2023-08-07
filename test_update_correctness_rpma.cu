#include "rpma.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>

using namespace std;


vector<string> datasets;


bool
test_update_correctness(thrust::host_vector<KEY_TYPE> &h_keys, \
	SIZE_TYPE node_num, SIZE_TYPE edge_num, SIZE_TYPE group_num) {

	RPMA rpma;
	init_rpma(rpma);

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

//SIZE_TYPE old_height = rpma.tree_height;
		update_rpma(rpma, keys, values, 0);
//if (rpma.tree_height != old_height) {
//cout << "fuck" << rpma.tree_height << endl;
//show_rpma(rpma);
//}
	}

	KEY_TYPE *rpma_keys;
	VALUE_TYPE *rpma_values;
	SIZE_TYPE compacted_size = get_data_rpma(rpma, rpma_keys, rpma_values);


	assert(edge_num == h_keys.size());
	if (compacted_size != edge_num) {
		cout << "size doesnt match, got " << compacted_size << ", but expect " << edge_num << endl;
		delete[] rpma_keys;
		delete[] rpma_values;
		return false;
	}

	thrust::sort(h_keys.begin(), h_keys.end());

	for (int i = 0; i < edge_num; ++i) {
		if (rpma_keys[i] != h_keys[i]) {
			cout << "data doesnt match at pos " << i << ", got " << rpma_keys[i] << ", but expect " << h_keys[i] << endl;
			delete[] rpma_keys;
			delete[] rpma_values;
			return false;
		}
	}

	delete[] rpma_keys;
	delete[] rpma_values;
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

	if (test_update_correctness(h_keys, node_num, edge_num, group_n))
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

	datasets.push_back("pokec");
	//datasets.push_back("cit");
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num, group_num);
	}
	return 0;
}
