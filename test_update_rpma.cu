#include "rpma.cuh"
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

//const KEY_TYPE SRC_SHIFT = 42;
//const KEY_TYPE DST_SHIFT = 20;
//const KEY_TYPE DST_AND_TIME_END = 0x000003FFFFEFFFFF;



bool
test_update(thrust::host_vector<KEY_TYPE> &h_keys, SIZE_TYPE node_num, SIZE_TYPE edge_num, SIZE_TYPE group_num,SIZE_TYPE showlpma) {

KEY_TYPE t = 0;

	RPMA rpma;
	init_csr_rpma(rpma, node_num);

	DEV_VEC_KEY keys;
	DEV_VEC_VALUE values;
	SIZE_TYPE std_group_size = (edge_num + group_num - 1) / group_num;
	int cnt = 0;
	int i;
	bool levelup;
	int level=0;
	//int ncount=0;
	DEV_VEC_SIZE d_query_srcs103;
	DEV_VEC_KEY d_query_constraint103;
	for (i = 0; i < 1000; ++i) {
		d_query_srcs103.push_back(100000+i*100);
		d_query_constraint103.push_back(1);
	}

	DEV_VEC_SIZE d_query_srcs104;
	DEV_VEC_KEY d_query_constraint104;
	for (i = 0; i < 10000; ++i) {
		d_query_srcs104.push_back(100000+i*10);
		d_query_constraint104.push_back(1);
	}
	DEV_VEC_KEY edgequery;
	DEV_VEC_VALUE edgequery_flag;
	for (i = 0; i < group_num; ++i) {
		DEV_VEC_SIZE update_nodes;
		int time=0;
		SIZE_TYPE bg = i * std_group_size;
		SIZE_TYPE ed = min((i + 1) * std_group_size, edge_num);
		if (cnt)
			break;
		if (ed == edge_num)
			++cnt;
		SIZE_TYPE group_size = ed - bg;
		values.resize(group_size, 1);
		keys.resize(group_size);
		
		edgequery.resize(group_size);
		edgequery_flag.resize(group_size,0);
		
		cErr(cudaMemcpy(RAW_PTR(edgequery), RAW_PTR(keys), sizeof(KEY_TYPE) * group_size, cudaMemcpyHostToDevice));
		
		cErr(cudaMemcpy(RAW_PTR(keys), RAW_PTR(h_keys) + bg, sizeof(KEY_TYPE) * group_size, cudaMemcpyHostToDevice));
		KEY_TYPE tt = 0;
		
		cErr(cudaDeviceSynchronize());
		TimeKeeper t1;
		levelup=update_rpma(rpma, keys, values,edgequery,edgequery_flag, 0);
		time=t1.checkTime("");
		cErr(cudaDeviceSynchronize());
		cout << "time cost this round,"<< time << endl;
		if(i>group_num/2){
		t += time;
		levelup=false;
		level+=1;
		}
	}

	cout << "Average: "<< t/level << endl;
	TimeKeeper t5;
	DEV_VEC_SIZE row_offset, col_index;
	cErr(cudaDeviceSynchronize());
		to_csr_rpma(rpma, row_offset, col_index);
	cErr(cudaDeviceSynchronize());
	cout << "CSR Query,"<< t5.checkTime("") << endl;
	if(showlpma==1){
	show_rpma(rpma);}
	KEY_TYPE *rpma_keys;
	VALUE_TYPE *rpma_values;
	SIZE_TYPE compacted_size = get_data_rpma(rpma, rpma_keys, rpma_values);

	thrust::host_vector<SIZE_TYPE> res;
	TimeKeeper t2;
	edge_query_rpma(res, rpma, edgequery,0);
	cout << "Edge Query,"<< t2.checkTime("") << endl;

	
	thrust::host_vector<SIZE_TYPE> offset;
	thrust::host_vector<SIZE_TYPE> len;
	thrust::host_vector<SIZE_TYPE> edgelist;
	thrust::host_vector<VALUE_TYPE> v_list;
	TimeKeeper t3;
	query_rpma(rpma, d_query_srcs103,d_query_constraint104, offset,len,edgelist,v_list);
	cout << "Neighbor Query 103: "<< t3.checkTime("") << endl;

	TimeKeeper t4;
	query_rpma(rpma, d_query_srcs104,d_query_constraint104, offset,len,edgelist,v_list);
	cout << "Neighbor Query 104: "<< t4.checkTime("") << endl;

	

	

	assert(edge_num == h_keys.size());
	if (compacted_size != edge_num+node_num) {
		cout << "size doesnt match, got " << compacted_size << ", but expect " << edge_num << endl;
		delete[] rpma_keys;
		delete[] rpma_values;
		return false;
	}
	cout << "size check done" << endl;
	thrust::sort(h_keys.begin(), h_keys.end());
	int j = 0;
	for (int i = 0; i < edge_num; ++i) {
	int s=(rpma_keys[j]>>SRC_SHIFT);
	int d=((rpma_keys[j]<<22)>>SRC_SHIFT);
	
	int su=(h_keys[i]>>SRC_SHIFT);
	int du=((h_keys[i]<<22)>>SRC_SHIFT);
	
	while((rpma_keys[j]<<SRC_SHIFT)==(DST_AND_TIME_END<<SRC_SHIFT)){
	//cout << "Show update     :" << s << "  " << d << endl;
		j++;
		s=(rpma_keys[j]>>SRC_SHIFT);
		d=((rpma_keys[j]<<22)>>SRC_SHIFT);
		}
		if (rpma_keys[j] != h_keys[i]) {
			cout << "round  :" << i << "  " << j << endl;
			cout << "Show rpma_keys  :" << s << "  " << d << endl;
			cout << "Show update     :" << su << "  " << du << endl;
			cout << "data doesnt match at pos " << i << ", got " << rpma_keys[j] << ", but expect " << h_keys[i] << endl;
			//delete[] rpma_keys;
			//delete[] rpma_values;
			return false;
		}
		j++;
	}
	cout << "edge check done" << endl;
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
		h_keys.push_back(src<<SRC_SHIFT | dst<<DST_SHIFT);
	}
	e_file.close();
}


void
test_all(string dataset, SIZE_TYPE edge_n, SIZE_TYPE group_n,SIZE_TYPE showlpma) {
	cout << "reading edgelist and timestamps" << endl;
	SIZE_TYPE node_num, edge_num;
	thrust::host_vector<KEY_TYPE> h_keys;
	read_edgelist(dataset, h_keys, node_num, edge_num, edge_n);
	
	group_n=edge_num/group_n;
	cout << "node num is:" << node_num << ", edge num is:" << edge_num <<", group num is:" << group_n << endl;

	if (test_update(h_keys, node_num, edge_num, group_n, showlpma))
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
	datasets.push_back(argv[3]);
	SIZE_TYPE showlpma= atoi(argv[4]);
	cErr(cudaDeviceSynchronize());
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (KEY_TYPE)2048*1024*1024));
	cErr(cudaDeviceSynchronize());

	
	//datasets.push_back("livejournal");
	//datasets.push_back("graph500");
	for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
		cout << "testing " << *dataset << endl;
		test_all(string("../edgelists/") + *dataset, edge_num, group_num, showlpma);
	}
	return 0;
}
