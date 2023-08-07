#include "gpma.cuh"
#include "util.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>

using namespace std;


vector<string> datasets;

const KEY_TYPE SRC_MASK = 0xFFFFFC0000000000;
const KEY_TYPE SRC_SHIFT = 42;
const KEY_TYPE DST_MASK = 0x000003FFFFF00000;
const KEY_TYPE DST_SHIFT = 20;
const KEY_TYPE TIME_MASK = 0x00000000000FFFFF;
const KEY_TYPE DST_AND_TIME_MASK = 0x000003FFFFFFFFFF;
const KEY_TYPE DST_AND_TIME_END = 0x000003FFFFEFFFFF;
const SIZE_TYPE NBR_END = 0x003FFFFE;

const SIZE_TYPE WINDOW_LENGTH = 100000;

//#define WITH_QUERY
#define TEST_GPMA
bool
test_query_and_update_with_timestamp_rpma(thrust::host_vector<KEY_TYPE> &h_keys, \
	vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, thrust::host_vector<SIZE_TYPE> &h_querys, \
	thrust::host_vector<KEY_TYPE> &h_constraints, SIZE_TYPE node_num, SIZE_TYPE edge_num, SIZE_TYPE query_num, \
	SIZE_TYPE group_num) {

	DEV_VEC_KEY keys;
	DEV_VEC_VALUE values;
	DEV_VEC_SIZE querys;
	DEV_VEC_KEY constraints;

	SIZE_TYPE std_group_size = (edge_num + group_num - 1) / group_num;

	thrust::host_vector<SIZE_TYPE> row_offset, nbrlist_len, edgelist;
	thrust::host_vector<VALUE_TYPE> vlist;
	
	row_offset.resize(1000001);
	nbrlist_len.resize(1000000);
	edgelist.resize(20000000);
	vlist.resize(20000000);

KEY_TYPE t = 0;
KEY_TYPE tq = 0;
	GPMA gpma;
	init_csr_gpma(gpma, node_num);

	for (SIZE_TYPE i = 0; i < group_num; ++i) {
		SIZE_TYPE bg = i * std_group_size;
		SIZE_TYPE ed = min((i + 1) * std_group_size, edge_num);
		if (ed <= bg) break;
		SIZE_TYPE group_size = ed - bg;

		SIZE_TYPE q_bg = (h_keys[bg] & TIME_MASK);
		SIZE_TYPE q_ed = (((h_keys[ed - 1] & TIME_MASK)) + 1) > query_num ? query_num : (((h_keys[ed - 1] & TIME_MASK)) + 1);
		SIZE_TYPE q_group_size = q_ed - q_bg;

		keys.resize(group_size);
		values.resize(group_size);
		cErr(cudaMemcpy(RAW_PTR(keys), RAW_PTR(h_keys) + bg, sizeof(KEY_TYPE) * group_size, cudaMemcpyHostToDevice));
		cErr(cudaMemset(RAW_PTR(values), 0, sizeof(VALUE_TYPE) * group_size));
		cErr(cudaDeviceSynchronize());
		
TimeKeeper t1;
cErr(cudaDeviceSynchronize());
		SIZE_TYPE low_thres = q_bg > WINDOW_LENGTH ? q_bg - WINDOW_LENGTH : 0;
		SIZE_TYPE thres = q_ed > q_bg ? low_thres : 0;
		update_gpma(gpma, keys, values);
cErr(cudaDeviceSynchronize());
t += t1.checkTime("");
	}
cout << t/group_num << "," << tq/group_num  << endl;
	return true;
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
read_querys(string &filename, thrust::host_vector<SIZE_TYPE> &h_querys, thrust::host_vector<KEY_TYPE> &h_constraints, \
	SIZE_TYPE &query_num, SIZE_TYPE query_n) {
	ifstream q_file;
	q_file.open((filename + ".query").c_str());

	q_file >> query_num;
	if (query_n)
		query_num = query_n;

	for (int i = 0; i < query_num; ++i) {
		KEY_TYPE src, lb, ub;
		q_file >> src;
		q_file >> lb;
		q_file >> ub;
		h_querys.push_back(src);
		h_constraints.push_back((lb << 32) | ub);
	}

	//cout << h_querys[999960] << "," << (h_constraints[999960] & 0xFFFFFFFF) << "," << (h_constraints[999960] >> 32) << endl;
	
	q_file.close();
}



void
test_all(string dataset, SIZE_TYPE edge_n, SIZE_TYPE query_n, SIZE_TYPE group_n) {
	cout << "reading edgelist and timestamps" << endl;
	SIZE_TYPE node_num, edge_num;
	thrust::host_vector<KEY_TYPE> h_keys;
	vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > G;
	read_edgelist_and_timestamps(dataset, h_keys, G, node_num, edge_num, edge_n);
	cout << "node num is:" << node_num << ", edge num is:" << edge_num << endl;

	cout << "reading queries" << endl;
	SIZE_TYPE query_num;
	thrust::host_vector<SIZE_TYPE> h_querys;
	thrust::host_vector<KEY_TYPE> h_constraints;
	read_querys(dataset, h_querys, h_constraints, query_num, query_n);
	cout << "query num is:" << query_num << endl;

	if (test_query_and_update_with_timestamp_rpma(h_keys, G, h_querys, h_constraints, \
		node_num, edge_num, query_num, group_n))
		cout << "query and update with timestamp test success" << endl;
	else {
		cout << "query and update with timestamp test fail" << endl;
		return;
	}
/*	
	init_rpma(rpma);

	DEV_VEC_KEY in_keys(h_keys.size() / 2);
	DEV_VEC_VALUE in_values(h_keys.size() / 2, 0);
	cErr(cudaMemcpy(RAW_PTR(in_keys), RAW_PTR(h_keys), h_keys.size() / 2 * sizeof(KEY_TYPE), cudaMemcpyHostToDevice));
	update_rpma(rpma, in_keys, in_values, 0);
	cErr(cudaMemcpy(RAW_PTR(in_keys), RAW_PTR(h_keys) + h_keys.size() / 2, h_keys.size() / 2 * sizeof(KEY_TYPE), cudaMemcpyHostToDevice));
	update_rpma(rpma, in_keys, in_values, 0);

//cout << rpma.tree_height << endl;
//show_rpma(rpma);
	KEY_TYPE *out_keys = NULL;
	VALUE_TYPE *out_values = NULL;
	get_data_rpma(rpma, out_keys, out_values);
	assert(out_keys != NULL && out_values != NULL);

	std::sort(h_keys.begin(), h_keys.end());

	for (int i = 0; i < h_keys.size(); ++i)
		if (h_keys[i] != out_keys[i]) {	
			cout << i << ": ";
			cout << h_keys[i] << "," << out_keys[i] << endl;
			if (i != h_keys.size() - 1)
				assert(h_keys[i] != h_keys[i+1]);
			cout << (h_keys[i]>>SRC_SHIFT) << "," << ((h_keys[i]&DST_MASK)>>DST_SHIFT) << "," << (h_keys[i]&TIME_MASK) << ",";
			cout << (out_keys[i]>>SRC_SHIFT) << "," << ((out_keys[i]&DST_MASK)>>DST_SHIFT) << "," << (out_keys[i]&TIME_MASK) << endl;
			return;
		}
	cout << "xixi" << endl;
*/
}


int
main(int args, char *argv[]) {

	SIZE_TYPE edge_num = atoi(argv[1]);
	SIZE_TYPE query_num = atoi(argv[2]);
	SIZE_TYPE group_num = atoi(argv[3]);

	cErr(cudaDeviceSynchronize());
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048*1024*1024));
	cErr(cudaDeviceSynchronize());

	datasets.push_back("pokec");
	datasets.push_back("livejournal");
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num, query_num, group_num);
	}
	return 0;
}
