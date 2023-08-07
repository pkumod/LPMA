#include "rpma.cuh"
#include "util.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>

using namespace std;

vector<string> datasets;

bool check_csr_result(vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, DEV_VEC_SIZE &d_row_offset, DEV_VEC_SIZE &d_col_index) {

	thrust::host_vector<SIZE_TYPE> row_offset(d_row_offset.size()), col_index(d_col_index.size());
	cErr(cudaMemcpy(RAW_PTR(row_offset), RAW_PTR(d_row_offset), d_row_offset.size() * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
	cErr(cudaMemcpy(RAW_PTR(col_index), RAW_PTR(d_col_index), d_col_index.size() * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));

	assert(G.size() + 1 == row_offset.size());

	for (int i = 0; i < G.size(); ++i) {
		SIZE_TYPE nbrlist_len = row_offset[i + 1] - row_offset[i] - 1;
		if (G[i].size() != nbrlist_len) {
			cout << " src " << i << ": size doesnt match" << endl;
			cout << "got " << nbrlist_len << ", but expect " << G[i].size() << endl;
			return false;
		}
		for (int j = 0; j < nbrlist_len; ++j) {
			if (col_index[row_offset[i] + j] != G[i][j].first) {
				cout << " src " << i << ": nbr doesnt match:" << j << endl;
				cout << "got " << col_index[row_offset[i] + j] << ", but expect " << G[i][j].first << endl;
				return false;
			}
		}
		assert(col_index[row_offset[i] + G[i].size()] == i);
	}
	return true;
}


bool
test_csr_rpma(thrust::host_vector<KEY_TYPE> &h_keys, \
	vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, SIZE_TYPE node_num, SIZE_TYPE edge_num) {

	RPMA rpma;
	init_csr_rpma(rpma, node_num);

	DEV_VEC_KEY keys(edge_num);
	DEV_VEC_VALUE values(edge_num);
	cErr(cudaMemcpy(RAW_PTR(keys), RAW_PTR(h_keys), sizeof(KEY_TYPE) * edge_num, cudaMemcpyHostToDevice));
	cErr(cudaMemset(RAW_PTR(values), 0, sizeof(VALUE_TYPE) * edge_num));
	cErr(cudaDeviceSynchronize());
		
	update_rpma(rpma, keys, values, 0);
	cErr(cudaDeviceSynchronize());

TimeKeeper t2;
cErr(cudaDeviceSynchronize());
	DEV_VEC_SIZE row_offset, col_index;
	to_csr_rpma(rpma, row_offset, col_index);
cErr(cudaDeviceSynchronize());
cout << t2.checkTime("") << endl;

	return check_csr_result(G, row_offset, col_index);
}

void
read_edgelist_and_timestamps(string &filename, thrust::host_vector<KEY_TYPE> &h_keys, \
	vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, SIZE_TYPE &node_num, SIZE_TYPE &edge_num, SIZE_TYPE edge_n) {
	ifstream e_file, t_file;
	e_file.open((filename + ".edgelist").c_str());
	t_file.open((filename + ".timestamps").c_str());

	e_file >> node_num;
	e_file >> edge_num;
	if (edge_n)
		edge_num = edge_n;

	G.resize(node_num);
	for (int i = 0; i < edge_num; ++i) {
		KEY_TYPE src, dst, time;
		e_file >> src;
		e_file >> dst;
		t_file >> time;

		assert(src < node_num && dst < node_num);
		G[src].push_back(make_pair(dst, time));
		h_keys.push_back((src << SRC_SHIFT)| (dst << DST_SHIFT) | time);
	}
	

	for (int i = 0; i < node_num; ++i) {
		sort(G[i].begin(), G[i].end());
	}

	e_file.close();
	t_file.close();
}

void
test_all(string dataset, SIZE_TYPE edge_n) {
	cout << "reading edgelist and timestamps" << endl;
	SIZE_TYPE node_num, edge_num;
	thrust::host_vector<KEY_TYPE> h_keys;
	vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > G;
	read_edgelist_and_timestamps(dataset, h_keys, G, node_num, edge_num, edge_n);
	cout << "node num is:" << node_num << ", edge num is:" << edge_num << endl;

	if (test_csr_rpma(h_keys, G, node_num, edge_num))
		cout << "csr test success" << endl;
	else {
		cout << "csr test fail" << endl;
		return;
	}
}


int
main(int args, char *argv[]) {

	SIZE_TYPE edge_num = atoi(argv[1]);

	cErr(cudaDeviceSynchronize());
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048*1024*1024));
	cErr(cudaDeviceSynchronize());

	datasets.push_back(argv[2]);
	//datasets.push_back("livejournal");
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num);
	}
	return 0;
}
