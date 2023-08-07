#include "rpma.cuh"
#include "util.cuh"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>

using namespace std;


int main(int argc, const char * argv[]) {

	if (argc < 3) {
		cout << "too few arguments" << endl;
		return 0;
	}
	if (argc > 4) {
		cout << "too many arguments" << endl;
	}
	int mode = 0;
	if (argc == 4)
		mode = (string(argv[3]) == "show");// 0 for cmp 1 for show
	SIZE_TYPE SIZE = atoi(argv[1]);
	SIZE_TYPE group_num = atoi(argv[2]);

	DEV_VEC_KEY keys;
	DEV_VEC_VALUE values;

	KEY_TYPE *h_keys = new KEY_TYPE[SIZE];
	VALUE_TYPE *h_values = new VALUE_TYPE[SIZE];
	ifstream in_key, in_value;
	in_key.open("keys.dat");
	in_value.open("values.dat");
	for (int i = 0; i < SIZE; ++i) {
		in_key >> h_keys[i];
		in_value >> h_values[i];
	}
	in_key.close();
	in_value.close();

	cout << "initiating rpma..." << endl;
	RPMA rpma;
	init_rpma(rpma);

	cout << "repeatly updating rpma..." << endl;
	SIZE_TYPE std_group_size = (SIZE + group_num - 1) / group_num;

	cErr(cudaDeviceSynchronize());
	cErr(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
	cErr(cudaDeviceSynchronize());

DEV_VEC_SIZE stat(32);
cErr(cudaMemset(RAW_PTR(stat), 0, sizeof(SIZE_TYPE)*32));
cErr(cudaDeviceSynchronize());

	TimeKeeper tk;
	for (int i = 0; i < group_num; ++i) {
		int bg = i * std_group_size;
		int ed = min((i + 1) * std_group_size, SIZE);
		int group_size = ed - bg;
		if (group_size <= 0) break;

		keys.resize(group_size);
		values.resize(group_size);
		cErr(cudaDeviceSynchronize());

		cErr(cudaMemcpy(RAW_PTR(keys), h_keys + bg, sizeof(KEY_TYPE) * group_size, cudaMemcpyHostToDevice));
		cErr(cudaMemcpy(RAW_PTR(values), h_values + bg, sizeof(VALUE_TYPE) * group_size, cudaMemcpyHostToDevice));
		cErr(cudaDeviceSynchronize());

	//	printf("update %d \n", i);
update_rpma(rpma, keys, values, stat);

		if (mode) {
			show_rpma(rpma);
		}
	}

unsigned long long sumstat = 0;
for (int i = 0; i < 32; ++i)
	sumstat += ((stat[i])<<i);
cout << sumstat;

	cErr(cudaDeviceSynchronize());
	tk.checkTime("update complete");

	cout << "generating gold keys and values..." << endl;
	thrust::sort_by_key(h_keys, h_keys + SIZE, h_values);
	cErr(cudaDeviceSynchronize());
		
	cout << "extracting data from rpma..." << endl;
	KEY_PTR o_keys;
	VALUE_PTR o_values;
	get_data_rpma(rpma, o_keys, o_values);

	if (mode) {
		/*
		cout << "keys:" << endl;
		for (int i = 0; i < SIZE; ++i) {
			cout << i << "\t" << h_keys[i] << "\t" << o_keys[i] << endl;
		}
		cout << "values:" << endl;
		for (int i = 0; i < SIZE; ++i) {
			cout << i << "\t" << h_values[i] << "\t" << o_values[i] << endl;
		}*/
		//show_rpma(rpma);
	}
	else {
//show_rpma(rpma);
		bool succ = true;
		for (int i = 0; i < SIZE; ++i) {
			if (h_keys[i] != o_keys[i] || h_values[i] != o_values[i]) {
				succ = false;
				SIZE_TYPE idx = GET_IDX(i, rpma.tree_height);
				SIZE_TYPE lane = GET_LANE(i, idx, rpma.tree_height);
				cout << "error in linear idx " << i << ", idx " << idx << ", lane " << lane << ", expect(" << h_keys[i] << ", " << h_values[i] << "), but got(" << o_keys[i] << ", " << o_values[i] << ")" << endl;
			}
		}
		if (succ) {
			cout << "test succeed" << endl;
		}
		else {
			cout << "test fail" << endl;
		}
	}

for (int i = 0; i < 32; ++i)
cout << stat[i] << endl;

	delete[] o_keys;
	delete[] o_values;

	destroy_rpma(rpma);
}
