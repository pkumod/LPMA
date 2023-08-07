#include "graph_analytics.cuh"
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

bool check_result(vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, SIZE_TYPE *offset, SIZE_TYPE *len, SIZE_TYPE *edgelist, \
		SIZE_TYPE *h_srcs, KEY_TYPE *h_constraints, SIZE_TYPE srcs_num) {
	for (int i = 0; i < srcs_num; ++i) {
		SIZE_TYPE src = h_srcs[i];
		SIZE_TYPE lb = h_constraints[i] >> 32;
		SIZE_TYPE ub = h_constraints[i] << 32 >> 32;
		vector<SIZE_TYPE> tmp;
		for (int j = 0; j < G[src].size(); ++j) {
			if (G[src][j].second >= lb && G[src][j].second <= ub)
				tmp.push_back(G[src][j].first);
		}
		if (tmp.size() != len[i]) {
			cout << "query " << i << " src " << src << ": size doesnt match" << endl;
			cout << "got " << len[i] << ", but expect " << tmp.size() << endl;
			cout << edgelist[offset[i]] << endl;
			return false;
		}
		for (int j = 0; j < len[i]; ++j) {
			if (edgelist[offset[i] + j] != tmp[j]) {
				cout << "query " << i << " src " << src << ": nbr doesnt match:" << j  << endl;
				cout << "got " << edgelist[offset[i] + j] << ", but expect " << tmp[j] << endl;
				return false;
			}
		}
	}
	return true;
}


bool check_edge_query_result(vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, SIZE_TYPE *res, KEY_TYPE *h_edge_querys, SIZE_TYPE query_num) {
	for (int i = 0; i < query_num; ++i) {
		KEY_TYPE key = h_edge_querys[i];
		SIZE_TYPE src = key >> SRC_SHIFT;
		SIZE_TYPE dst = (key & DST_MASK) >> DST_SHIFT;
		SIZE_TYPE lb = ((key & TIME_MASK) > WINDOW_LENGTH) ? ((key & TIME_MASK) - WINDOW_LENGTH) : 0;
		SIZE_TYPE ub = key & TIME_MASK;
		vector<SIZE_TYPE> tmp;
		for (int j = 0; j < G[src].size(); ++j) {
			if (G[src][j].second >= lb && G[src][j].second <= ub && G[src][j].first == dst)
				tmp.push_back(G[src][j].first);
		}
		if (tmp.size() != res[i]) {
			cout << "query " << i << " src " << src << ": size doesnt match" << endl;
			cout << "got " << res[i] << ", but expect " << tmp.size() << endl;
			return false;
		}
	}
	return true;
}

#define WITH_QUERY
#define WITH_ANALYTICS
//#define WITH_EDGE_QUERY

bool
test_query_and_update_with_timestamp_rpma(thrust::host_vector<KEY_TYPE> &h_keys, \
	vector<vector<pair<SIZE_TYPE, SIZE_TYPE> > > &G, thrust::host_vector<SIZE_TYPE> &h_querys, \
	thrust::host_vector<KEY_TYPE> &h_constraints, thrust::host_vector<KEY_TYPE> &h_edge_querys, \
	SIZE_TYPE node_num, SIZE_TYPE edge_num, SIZE_TYPE query_num, SIZE_TYPE group_num) {

	DEV_VEC_KEY keys;
	DEV_VEC_VALUE values;
	DEV_VEC_SIZE querys;
	DEV_VEC_KEY constraints;
	DEV_VEC_KEY edge_querys;

	SIZE_TYPE std_group_size = (edge_num + group_num - 1) / group_num;

	thrust::host_vector<SIZE_TYPE> row_offset, nbrlist_len, edgelist;
	thrust::host_vector<VALUE_TYPE> vlist;
	row_offset.resize(1000001);
	nbrlist_len.resize(1000000);
	edgelist.resize(20000000);
	vlist.resize(20000000);

	thrust::host_vector<SIZE_TYPE> res;
	res.resize(std_group_size, 0);

int t = 0, tq = 0, ta = 0, tc = 0;
	RPMA rpma;
	init_csr_rpma(rpma, node_num);

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
#ifdef WITH_QUERY
		SIZE_TYPE low_thres = q_bg > WINDOW_LENGTH ? q_bg - WINDOW_LENGTH : 0;
		SIZE_TYPE thres = q_ed > q_bg ? low_thres : 0;
#else
		SIZE_TYPE thres = 0;
#endif
		update_rpma(rpma, keys, values, thres);
cErr(cudaDeviceSynchronize());
t += t1.checkTime("");


#ifdef WITH_QUERY
		if (q_ed > q_bg) {
			querys.resize(q_group_size);
			constraints.resize(q_group_size);
			cErr(cudaMemcpy(RAW_PTR(querys), RAW_PTR(h_querys) + q_bg, sizeof(SIZE_TYPE) * q_group_size, cudaMemcpyHostToDevice));
			cErr(cudaMemcpy(RAW_PTR(constraints), RAW_PTR(h_constraints) + q_bg, sizeof(KEY_TYPE) * q_group_size, cudaMemcpyHostToDevice));
			cErr(cudaDeviceSynchronize());
TimeKeeper t2;
cErr(cudaDeviceSynchronize());
			query_rpma(rpma, querys, constraints, row_offset, nbrlist_len, edgelist, vlist);
cErr(cudaDeviceSynchronize());
tq += t2.checkTime("");
			
			cout << rpma.tree_height << endl;
			bool res = check_result(G, RAW_PTR(row_offset), RAW_PTR(nbrlist_len), RAW_PTR(edgelist), \
					RAW_PTR(h_querys) + q_bg, RAW_PTR(h_constraints) + q_bg, q_group_size);
			cout << (res ? "succeed" : "fail") << endl;	
			if (!res)
				return false;

		}
#endif

#ifdef WITH_EDGE_QUERY
		if (q_ed > q_bg) {
			edge_querys.resize(q_group_size);
			cErr(cudaMemcpy(RAW_PTR(edge_querys), RAW_PTR(h_edge_querys) + q_bg, sizeof(KEY_TYPE) * q_group_size, cudaMemcpyHostToDevice));
			cErr(cudaDeviceSynchronize());
TimeKeeper t2;
cErr(cudaDeviceSynchronize());
			edge_query_rpma(rpma, edge_querys, res);
cErr(cudaDeviceSynchronize());
tq += t2.checkTime("");

			bool check = check_edge_query_result(G, RAW_PTR(res), RAW_PTR(h_edge_querys) + q_bg, q_group_size);
			cout << (check ? "succeed" : "fail") << endl;	
			if (!check)
				return false;		
		}
#endif

#ifdef WITH_ANALYTICS

TimeKeeper t3;
cErr(cudaDeviceSynchronize());
		DEV_VEC_SIZE row_offset, col_index;
		to_csr_rpma(rpma, row_offset, col_index);
cErr(cudaDeviceSynchronize());
tc += t3.checkTime("");

		int *off_h = new int[row_offset.size()], *idx_h = new int[col_index.size()];
		cErr(cudaMemcpy(off_h, RAW_PTR(row_offset), sizeof(int) * row_offset.size(), cudaMemcpyDeviceToHost));
		cErr(cudaMemcpy(idx_h, RAW_PTR(col_index), sizeof(int) * col_index.size(), cudaMemcpyDeviceToHost));
		bfs(row_offset.size(), col_index.size(), off_h, idx_h, ta);
		delete[] off_h;
		delete[] idx_h;
#endif

	}
cout << t/group_num << "," << tq/group_num  << "," << tc/group_num << "," << ta/group_num << endl;
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
read_queries(string &filename, thrust::host_vector<SIZE_TYPE> &h_querys, thrust::host_vector<KEY_TYPE> &h_constraints, \
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
	
	q_file.close();
}


void
read_edge_queries(string &filename, thrust::host_vector<KEY_TYPE> &h_edge_queries, SIZE_TYPE &edge_query_num, SIZE_TYPE edge_query_n) {
	ifstream q_file;
	q_file.open((filename + ".edgequery").c_str());

	q_file >> edge_query_num;
	if (edge_query_n)
		edge_query_num = edge_query_n;

	for (KEY_TYPE i = 0; i < edge_query_num; ++i) {
		KEY_TYPE src, dst;
		q_file >> src;
		q_file >> dst;
		h_edge_queries.push_back((src << SRC_SHIFT) | (dst << DST_SHIFT) | i);
	}
	
	q_file.close();
}


void
test_all(string dataset, SIZE_TYPE edge_n, SIZE_TYPE query_n, SIZE_TYPE edge_query_n, SIZE_TYPE group_n) {
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
	read_queries(dataset, h_querys, h_constraints, query_num, query_n);
	cout << "query num is:" << query_num << endl;

	cout << "reading edge queries" << endl;
	SIZE_TYPE edge_query_num;
	thrust::host_vector<KEY_TYPE> h_edge_queries;
	read_edge_queries(dataset, h_edge_queries, edge_query_num, edge_query_n);
	cout << "edge query num is:" << edge_query_num << endl;

	if (test_query_and_update_with_timestamp_rpma(h_keys, G, h_querys, h_constraints, h_edge_queries, \
		node_num, edge_num, query_num, group_n))
		cout << "query and update with timestamp test success" << endl;
	else {
		cout << "query and update with timestamp test fail" << endl;
		return;
	}
}


int
main(int args, char *argv[]) {

	SIZE_TYPE edge_num = atoi(argv[1]);
	SIZE_TYPE query_num = atoi(argv[2]);
	SIZE_TYPE edge_query_num = atoi(argv[3]);
	SIZE_TYPE group_num = atoi(argv[4]);

	cErr(cudaDeviceSynchronize());
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048*1024*1024));
	cErr(cudaDeviceSynchronize());

	datasets.push_back("pokec");
	datasets.push_back("livejournal");
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num, query_num, edge_query_num, group_num);
	}
	return 0;
}
