#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <cassert>
#include <stdio.h>
#include <algorithm>
#include "util.cuh"

#define cErr(errcode) { gpuAssert((errcode), __FILE__, __LINE__); }
__inline__ __host__ __device__
void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

typedef unsigned long long KEY_TYPE;
typedef unsigned char VALUE_TYPE;
typedef unsigned int SIZE_TYPE;

typedef thrust::device_vector<KEY_TYPE> DEV_VEC_KEY;
typedef thrust::device_vector<VALUE_TYPE> DEV_VEC_VALUE;
typedef thrust::device_vector<SIZE_TYPE> DEV_VEC_SIZE;

typedef KEY_TYPE* KEY_PTR;
typedef VALUE_TYPE* VALUE_PTR;

typedef thrust::device_vector<KEY_PTR> DEV_VEC_LEVEL_KEY_PTR;
typedef thrust::device_vector<VALUE_PTR> DEV_VEC_LEVEL_VALUE_PTR;

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())
#define NB 1
#define NTPB 1024
const KEY_TYPE KEY_NONE = -1;
const KEY_TYPE KEY_MAX = -2;
const SIZE_TYPE SIZE_NONE = -1;
const VALUE_TYPE VALUE_NONE = -1;

const KEY_TYPE SRC_MASK = 0xFFFFFC0000000000;
const KEY_TYPE SRC_SHIFT = 42;
const KEY_TYPE DST_MASK = 0x000003FFFFF00000;
const KEY_TYPE DST_SHIFT = 20;
const KEY_TYPE TIME_MASK = 0x00000000000FFFFF;
const KEY_TYPE DST_AND_TIME_MASK = 0x000003FFFFFFFFFF;
const KEY_TYPE DST_AND_TIME_END = 0x000003FFFFEFFFFF;
const SIZE_TYPE NBR_END = 0x003FFFFE;

const SIZE_TYPE WINDOW_LENGTH = 100000;

//pre allocated data size
const unsigned long MAX_UPDATE_WIDTH = 32L << 22;

const SIZE_TYPE MAX_BLOCKS_NUM = 96 * 8;
#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) min(MAX_BLOCKS_NUM, (CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1)

class RPMA {
public:
	/*
    DEV_VEC_KEY keys;
    DEV_VEC_VALUE values;
    */

	DEV_VEC_LEVEL_KEY_PTR levels_key_ptr_array;
	DEV_VEC_LEVEL_VALUE_PTR levels_value_ptr_array;

    SIZE_TYPE segment_length;
    SIZE_TYPE tree_height;
	
	SIZE_TYPE total_items;
    bool reachroot=false;

	double density_lower_thres_leaf = 0.08;
    double density_lower_thres_root = 0.42;
    double density_upper_thres_root = 1;
    double density_upper_thres_leaf = 1;

    thrust::host_vector<SIZE_TYPE> lower_element;
    thrust::host_vector<SIZE_TYPE> upper_element;

    SIZE_TYPE row_num;
    DEV_VEC_SIZE csr_idx;
    DEV_VEC_SIZE csr_lane;

	//pre alloctated structures for rebalancing kernel
	SIZE_TYPE *compacted_size;
	KEY_TYPE *tmp_keys;
	VALUE_TYPE *tmp_values;
	SIZE_TYPE *tmp_exscan;
    SIZE_TYPE *tmp_label;
	KEY_TYPE *tmp_keys_sorted;
	VALUE_TYPE *tmp_values_sorted;
};

__forceinline__ __host__ __device__
SIZE_TYPE fls(SIZE_TYPE x) {
    SIZE_TYPE r = 32;
    if (!x)
        return 0;
    if (!(x & 0xffff0000u))
        x <<= 16, r -= 16;
    if (!(x & 0xff000000u))
        x <<= 8, r -= 8;
    if (!(x & 0xf0000000u))
        x <<= 4, r -= 4;
    if (!(x & 0xc0000000u))
        x <<= 2, r -= 2;
    if (!(x & 0x80000000u))
        x <<= 1, r -= 1;
    return r;
}

__forceinline__ __host__ __device__
SIZE_TYPE lls(SIZE_TYPE x) {
    SIZE_TYPE r = 1;
    if (!(x & 0x0000ffffu))
        x >>= 16, r += 16;
    if (!(x & 0x000000ffu))
        x >>= 8, r += 8;
    if (!(x & 0x0000000fu))
        x >>= 4, r += 4;
    if (!(x & 0x00000003u))
        x >>= 2, r += 2;
    if (!(x & 0x00000001u))
        x >>= 1, r += 1;
    return r;
}

#define GET_IDX(LINEAR_IDX, HEIGHT) (HEIGHT + 1 - lls((LINEAR_IDX >> 5) + 1))
#define GET_LANE(LINEAR_IDX, IDX, HEIGHT) ((((((LINEAR_IDX >> 5) + 1) >> (HEIGHT - IDX)) - 1) >> 1) << 5) + (LINEAR_IDX % 32)
#define GET_LINEAR_IDX(IDX, LANE, HEIGHT) ((32 << (HEIGHT- IDX)) - 32 + ((LANE >> 5) << (HEIGHT - IDX + 1) << 5))

__host__
void recalculate_density(RPMA &rpma) {
    rpma.lower_element.resize(26);
    rpma.upper_element.resize(26);
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE level_length = rpma.segment_length;
    for (SIZE_TYPE i = 0; i <= 25; i++) {
/*
        double density_lower = rpma.density_lower_thres_root
                + (rpma.density_lower_thres_leaf - rpma.density_lower_thres_root) * (rpma.tree_height - i)
                        / rpma.tree_height;
        double density_upper = rpma.density_upper_thres_root
                + (rpma.density_upper_thres_leaf - rpma.density_upper_thres_root) * (rpma.tree_height - i)
                        / rpma.tree_height;

	double density_lower = 0;
	double density_upper = 1;
        rpma.lower_element[i] = (SIZE_TYPE) ceil(density_lower * level_length);
        rpma.upper_element[i] = (SIZE_TYPE) floor(density_upper * level_length);

        // special trim for wrong threshold introduced by float-integer conversion
        if (0 < i) {
            rpma.lower_element[i] = max(rpma.lower_element[i], 2 * rpma.lower_element[i - 1]);
            rpma.upper_element[i] = min(rpma.upper_element[i], 2 * rpma.upper_element[i - 1]);
        }
        level_length <<= 1;
*/
	rpma.lower_element[i] = rpma.density_lower_thres_root * level_length;
	rpma.upper_element[i] = level_length;
	level_length <<= 1;
    }
}

template<typename T>
__global__
void memcpy_kernel(T *dest, const T *src, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        dest[i] = src[i];
    }
}

template<typename T>
__global__
void memset_kernel(T *data, T value, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        data[i] = value;
    }
}

template<typename T>
__global__
void level_memset_kernel(T* data[], T value, SIZE_TYPE size, SIZE_TYPE data_offset, SIZE_TYPE tree_height) {
	SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id + data_offset; i < size + data_offset; i += block_offset) {
    	SIZE_TYPE idx = GET_IDX(i, tree_height);
    	SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
        data[idx][lane] = value;
    }
}


__device__
void cub_sort_key_value(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_PTR tmp_keys,
        VALUE_PTR tmp_values, SIZE_TYPE update_node) {
	
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaDeviceSynchronize());

    cErr(cudaFree(d_temp_storage));  
}

__global__
void global_cub_sort_key_value(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_PTR tmp_keys,
        VALUE_PTR tmp_values, SIZE_TYPE update_node) {

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaDeviceSynchronize());

    cErr(cudaFree(d_temp_storage));
}

__device__
SIZE_TYPE handle_del_mod(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE seg_length,
	KEY_TYPE key, VALUE_TYPE value, SIZE_TYPE leaf, SIZE_TYPE idx, SIZE_TYPE lane) {

    if (VALUE_NONE == value)
        leaf = SIZE_NONE;
    for (SIZE_TYPE i = 0; i < seg_length; i++) {
        if (keys[idx][lane + i] == key) {
            values[idx][lane + i] = value;
            leaf = SIZE_NONE;
            break;
        }
    }
    return leaf;
}

__global__
void locate_leaf_kernel(KEY_PTR key_ptrs[], VALUE_PTR value_ptrs[], SIZE_TYPE seg_length,
	SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
	SIZE_TYPE *leaf) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    __shared__ KEY_PTR block_key_ptrs[26];
    __shared__ VALUE_PTR block_value_ptrs[26];
    __shared__ KEY_TYPE levels_key[32 * ((1<<5)-1)];	
    __shared__ VALUE_TYPE levels_value[32 * ((1<<5)-1)];	

	if (threadIdx.x == 0) {
		int thres = (tree_height > 4 ? 4 : tree_height);
		for (int i = 0; i <= thres; ++i) {
			int offset = (32 << i) - 32;
			memcpy(levels_key + offset, key_ptrs[i], sizeof(KEY_TYPE) * (32 << i));
			memcpy(levels_value + offset, value_ptrs[i], sizeof(VALUE_TYPE) * (32 << i));
			block_key_ptrs[i] = levels_key + offset;
			block_value_ptrs[i] = levels_value + offset;
		}
		memcpy(block_key_ptrs + thres + 1, key_ptrs + thres + 1, sizeof(KEY_PTR) * (25 - thres));
		memcpy(block_value_ptrs + thres + 1, value_ptrs + thres + 1, sizeof(VALUE_PTR) * (25 - thres));
	}

    __syncthreads();

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
 
        KEY_TYPE key = update_keys[i];
        VALUE_TYPE value = update_values[i];
    	SIZE_TYPE idx = 0, lane = 0;
    	while(idx < tree_height && KEY_NONE != block_key_ptrs[idx][lane]) {

    		if (block_key_ptrs[idx][lane] > key) {
    			++idx;
    			lane <<= 1;
    			continue;
    		}
    		
    		SIZE_TYPE l = 0, r = seg_length - 1;
    		while(l < r) {
    			SIZE_TYPE mid  = (l + r + 1) >> 1;
    			if (block_key_ptrs[idx][lane + mid] == KEY_NONE)
    				r = mid - 1;
    			else
    				l = mid;
    		}			
    		
    		if (key > block_key_ptrs[idx][lane + l]) {
    			++idx;
                    	lane = (lane << 1) + seg_length;
    		}
    		else
    			break;			

    	}
    	//SIZE_TYPE prefix = (seg_length << (tree_height - idx)) - seg_length + (lane << (tree_height - idx + 1));
        SIZE_TYPE prefix = GET_LINEAR_IDX(idx, lane, tree_height);
        //prefix = handle_del_mod(block_key_ptrs, block_value_ptrs, seg_length, key, value, prefix, idx, lane);
        leaf[i] = prefix;
    }
}

__global__
void locate_leaf_kernel_mixed(KEY_PTR key_ptrs[], VALUE_PTR value_ptrs[], SIZE_TYPE seg_length,
	SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, KEY_TYPE *edgequery, VALUE_TYPE *edgequery_flag, SIZE_TYPE update_size,SIZE_TYPE edgequery_size,
	SIZE_TYPE *leaf) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    __shared__ KEY_PTR block_key_ptrs[26];
    __shared__ VALUE_PTR block_value_ptrs[26];
    __shared__ KEY_TYPE levels_key[32 * ((1<<5)-1)];	
    __shared__ VALUE_TYPE levels_value[32 * ((1<<5)-1)];	

	if (threadIdx.x == 0) {
		int thres = (tree_height > 4 ? 4 : tree_height);
		for (int i = 0; i <= thres; ++i) {
			int offset = (32 << i) - 32;
			memcpy(levels_key + offset, key_ptrs[i], sizeof(KEY_TYPE) * (32 << i));
			memcpy(levels_value + offset, value_ptrs[i], sizeof(VALUE_TYPE) * (32 << i));
			block_key_ptrs[i] = levels_key + offset;
			block_value_ptrs[i] = levels_value + offset;
		}
		memcpy(block_key_ptrs + thres + 1, key_ptrs + thres + 1, sizeof(KEY_PTR) * (25 - thres));
		memcpy(block_value_ptrs + thres + 1, value_ptrs + thres + 1, sizeof(VALUE_PTR) * (25 - thres));
	}

    __syncthreads();

    for (SIZE_TYPE i = global_thread_id; i < update_size + edgequery_size; i += block_offset) {
	if(i<update_size){
        KEY_TYPE key = update_keys[i];
        VALUE_TYPE value = update_values[i];
    	SIZE_TYPE idx = 0, lane = 0;
    	while(idx < tree_height && KEY_NONE != block_key_ptrs[idx][lane]) {

    		if (block_key_ptrs[idx][lane] > key) {
    			++idx;
    			lane <<= 1;
    			continue;
    		}
    		
    		SIZE_TYPE l = 0, r = seg_length - 1;
    		while(l < r) {
    			SIZE_TYPE mid  = (l + r + 1) >> 1;
    			if (block_key_ptrs[idx][lane + mid] == KEY_NONE)
    				r = mid - 1;
    			else
    				l = mid;
    		}			
    		
    		if (key > block_key_ptrs[idx][lane + l]) {
    			++idx;
                    	lane = (lane << 1) + seg_length;
    		}
    		else
    			break;			

    	}
    	//SIZE_TYPE prefix = (seg_length << (tree_height - idx)) - seg_length + (lane << (tree_height - idx + 1));
        SIZE_TYPE prefix = GET_LINEAR_IDX(idx, lane, tree_height);
        //prefix = handle_del_mod(block_key_ptrs, block_value_ptrs, seg_length, key, value, prefix, idx, lane);
        leaf[i] = prefix;
		}else{
			KEY_TYPE key = edgequery[i-update_size];
			VALUE_TYPE value = edgequery_flag[i-update_size];
			SIZE_TYPE idx = 0, lane = 0;
			while(idx < tree_height && KEY_NONE != block_key_ptrs[idx][lane]) {

				if (block_key_ptrs[idx][lane] > key) {
					++idx;
					lane <<= 1;
					continue;
				}
				
				SIZE_TYPE l = 0, r = seg_length - 1;
				while(l < r) {
					SIZE_TYPE mid  = (l + r + 1) >> 1;
					if (block_key_ptrs[idx][lane + mid] == KEY_NONE)
						r = mid - 1;
					else
						l = mid;
				}			
				
				if (key > block_key_ptrs[idx][lane + l]) {
					++idx;
							lane = (lane << 1) + seg_length;
				}
				else{
					
					edgequery_flag[i-update_size]=1;
					break;	}		

			}
		}
    }
}
__host__
void locate_leaf_batch(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE seg_length,
        SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
        SIZE_TYPE *leaf) {

    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);

    locate_leaf_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keys, values, seg_length, tree_height, update_keys,
            update_values, update_size, leaf);
    cErr(cudaDeviceSynchronize());
}

__host__
void locate_leaf_batch_mixed(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE seg_length,
        SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,KEY_TYPE *edgequery, VALUE_TYPE *edgequery_flag, SIZE_TYPE edgequery_size,
        SIZE_TYPE *leaf) {

    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size + edgequery_size);

    locate_leaf_kernel_mixed<<<BLOCKS_NUM, THREADS_NUM>>>(keys, values, seg_length, tree_height, update_keys,
            update_values, edgequery, edgequery_flag, update_size, edgequery_size, leaf);
    cErr(cudaDeviceSynchronize());
}

struct three_tuple_first_none {
    typedef thrust::tuple<SIZE_TYPE, KEY_TYPE, VALUE_TYPE> Tuple;
    __host__ __device__
    bool operator()(const Tuple &a) {
        return SIZE_NONE == thrust::get<0>(a);
    }
};
__host__
void compact_insertions(DEV_VEC_SIZE &update_nodes, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values,
        SIZE_TYPE &update_size) {

    auto zip_begin = thrust::make_zip_iterator(
            thrust::make_tuple(update_nodes.begin(), update_keys.begin(), update_values.begin()));
    auto zip_end = thrust::remove_if(zip_begin, zip_begin + update_size, three_tuple_first_none());
    cErr(cudaDeviceSynchronize());
    update_size = zip_end - zip_begin;
}

__host__ SIZE_TYPE group_insertion_by_node(SIZE_TYPE *update_nodes, SIZE_TYPE update_size,
        SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset) {

    // step1: encode
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    SIZE_TYPE *tmp_offset;
    cErr(cudaMalloc(&tmp_offset, sizeof(SIZE_TYPE) * update_size));

    SIZE_TYPE *num_runs_out;
    cErr(cudaMalloc(&num_runs_out, sizeof(SIZE_TYPE)));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, update_nodes,
        unique_update_nodes, tmp_offset, num_runs_out, update_size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, update_nodes,
        unique_update_nodes, tmp_offset, num_runs_out, update_size));
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE unique_node_size[1];
    cErr(cudaMemcpy(unique_node_size, num_runs_out, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(num_runs_out));
    cErr(cudaFree(d_temp_storage));

    // step2: exclusive scan
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp_offset,
            update_offset, unique_node_size[0]));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp_offset,
            update_offset, unique_node_size[0]));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    cErr(cudaMemcpy(update_offset + unique_node_size[0], &update_size, sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(tmp_offset));

    return unique_node_size[0];
}

__host__
void compress_insertions_by_node(DEV_VEC_SIZE &update_nodes, SIZE_TYPE update_size,
        DEV_VEC_SIZE &unique_update_nodes, DEV_VEC_SIZE &update_offset, SIZE_TYPE &unique_node_size) {
    unique_node_size = group_insertion_by_node(RAW_PTR(update_nodes), update_size, RAW_PTR(unique_update_nodes),
            RAW_PTR(update_offset));
    cErr(cudaDeviceSynchronize());
}

struct kv_tuple_none {
    typedef thrust::tuple<KEY_TYPE, VALUE_TYPE> Tuple;
    __host__ __device__
    bool operator()(const Tuple &a) {
        return KEY_NONE == thrust::get<0>(a) || VALUE_NONE == thrust::get<1>(a);
    }
};


template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__
void block_compact_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE &compacted_size,
	KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE thres) {
    typedef cub::BlockScan<SIZE_TYPE, THREAD_PER_BLOCK> BlockScan;
    SIZE_TYPE thread_id = threadIdx.x;

    KEY_TYPE *block_keys = keys;
    VALUE_TYPE *block_values = values;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];

    SIZE_TYPE thread_offset = thread_id * ITEM_PER_THREAD;
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_keys[i] = block_keys[i];
        thread_values[i] = block_values[i];
        //block_keys[i] = KEY_NONE;
    }

    __shared__ typename BlockScan::TempStorage temp_storage;
    SIZE_TYPE thread_data[ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_data[i] = (thread_keys[i] == KEY_NONE || (thread_keys[i] & TIME_MASK) < (KEY_TYPE)thres) ? 0 : 1 ;
    }
    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    __shared__ SIZE_TYPE exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        exscan[i + thread_offset] = thread_data[i];
    }
    __syncthreads();

    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        if (thread_id == THREAD_PER_BLOCK - 1 && i == ITEM_PER_THREAD - 1)
            continue;
        if (exscan[thread_offset + i] != exscan[thread_offset + i + 1]) {
            SIZE_TYPE loc = exscan[thread_offset + i];
            tmp_keys[loc] = thread_keys[i];
            tmp_values[loc] = thread_values[i];
        }
    }

    // special logic for the last element
    if (thread_id == THREAD_PER_BLOCK - 1) {
        SIZE_TYPE loc = exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD - 1];
        if (thread_keys[ITEM_PER_THREAD - 1] == KEY_NONE || (thread_keys[ITEM_PER_THREAD - 1] & TIME_MASK) < (KEY_TYPE)thres) {
            compacted_size = loc;
        } else {
            compacted_size = loc + 1;
            tmp_keys[loc] = thread_keys[ITEM_PER_THREAD - 1];
            tmp_values[loc] = thread_values[ITEM_PER_THREAD - 1];
        }
    }
    __syncthreads();

    for (SIZE_TYPE i = compacted_size + thread_id; i < THREAD_PER_BLOCK * ITEM_PER_THREAD; i += THREAD_PER_BLOCK) {
    	tmp_keys[i] = KEY_NONE;
    }
}

template<typename FIRST_TYPE, typename SECOND_TYPE>
__device__
void block_pair_copy_kernel(FIRST_TYPE *dest_first, SECOND_TYPE *dest_second, FIRST_TYPE *src_first,
        SECOND_TYPE *src_second, SIZE_TYPE size) {
    for (SIZE_TYPE i = threadIdx.x; i < size; i += blockDim.x) {
        dest_first[i] = src_first[i];
        dest_second[i] = src_second[i];
    }
}

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__
void block_redispatch_kernel(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE rebalance_width, SIZE_TYPE seg_length,
        SIZE_TYPE merge_size, SIZE_TYPE update_node, KEY_PTR block_keys, VALUE_PTR block_values,
        SIZE_TYPE block_offset, SIZE_TYPE tree_height, SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane) {

    // step2: sort by key with value on shared memory
    typedef cub::BlockLoad<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockKeyLoadT;
    typedef cub::BlockLoad<VALUE_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockValueLoadT;
    typedef cub::BlockStore<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockKeyStoreT;
    typedef cub::BlockStore<VALUE_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockValueStoreT;
    typedef cub::BlockRadixSort<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, VALUE_TYPE> BlockRadixSortT;

    __shared__ union {
            typename BlockKeyLoadT::TempStorage key_load;
            typename BlockValueLoadT::TempStorage value_load;
            typename BlockKeyStoreT::TempStorage key_store;
            typename BlockValueStoreT::TempStorage value_store;
            typename BlockRadixSortT::TempStorage sort;
        } temp_storage;

        KEY_TYPE thread_keys[ITEM_PER_THREAD];
        VALUE_TYPE thread_values[ITEM_PER_THREAD];
        BlockKeyLoadT(temp_storage.key_load).Load(block_keys, thread_keys);
        BlockValueLoadT(temp_storage.value_load).Load(block_values, thread_values);
        __syncthreads();

        BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
        __syncthreads();

        BlockKeyStoreT(temp_storage.key_store).Store(block_keys, thread_keys);
        BlockValueStoreT(temp_storage.value_store).Store(block_values, thread_values);
        __syncthreads();

        // step3: evenly re-dispatch KVs to leaf segments
        KEY_TYPE frac = rebalance_width / seg_length;
        KEY_TYPE deno = merge_size;
        for (SIZE_TYPE i = threadIdx.x + block_offset; i < rebalance_width + block_offset; i += blockDim.x) {
            SIZE_TYPE idx = GET_IDX(i, tree_height);
            SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
            keys[idx][lane] = KEY_NONE;
        }
        __syncthreads();
        for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
            SIZE_TYPE seg_idx = (SIZE_TYPE) (frac * i / deno);
            SIZE_TYPE seg_lane = (SIZE_TYPE) (frac * i % deno / frac);
            SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;

            KEY_TYPE cur_key = block_keys[i];
            VALUE_TYPE cur_value = block_values[i];
            SIZE_TYPE linear_idx = proj_location + block_offset;
            SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
            SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);
            keys[idx][lane] = cur_key;
            values[idx][lane] = cur_value;

            //addition for csr
            if ((cur_key & DST_AND_TIME_MASK) == DST_AND_TIME_END) {
                SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> SRC_SHIFT);
                csr_idx[cur_row + 1] = idx;
                csr_lane[cur_row + 1] = lane;
            }
        }
    }

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__
void block_redispatch_merge_kernel(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE rebalance_width, SIZE_TYPE seg_length,
        SIZE_TYPE merge_size, SIZE_TYPE update_node, KEY_PTR block_keys, VALUE_PTR block_values,
        SIZE_TYPE block_offset, SIZE_TYPE tree_height, SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane) {

        // step3: evenly re-dispatch KVs to leaf segments
        KEY_TYPE frac = rebalance_width / seg_length;
        KEY_TYPE deno = merge_size;
        for (SIZE_TYPE i = threadIdx.x + block_offset; i < rebalance_width + block_offset; i += blockDim.x) {
            SIZE_TYPE idx = GET_IDX(i, tree_height);
            SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
            keys[idx][lane] = KEY_NONE;
        }
        __syncthreads();
        for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
            SIZE_TYPE seg_idx = (SIZE_TYPE) (frac * i / deno);
            SIZE_TYPE seg_lane = (SIZE_TYPE) (frac * i % deno / frac);
            SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;

            KEY_TYPE cur_key = block_keys[i];
            VALUE_TYPE cur_value = block_values[i];
            SIZE_TYPE linear_idx = proj_location + block_offset;
            SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
            SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);
            keys[idx][lane] = cur_key;
            values[idx][lane] = cur_value;

            //addition for csr
            if ((cur_key & DST_AND_TIME_MASK) == DST_AND_TIME_END) {
                SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> SRC_SHIFT);
                csr_idx[cur_row + 1] = idx;
                csr_lane[cur_row + 1] = lane;
            }
        }
    }
template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__ 
void mergesmallk(KEY_PTR M, VALUE_PTR Mvalue, SIZE_TYPE merge_size, KEY_TYPE *A, VALUE_TYPE *Avalue, KEY_TYPE startA,SIZE_TYPE lenA,KEY_TYPE *B,VALUE_TYPE* Bvalue,KEY_TYPE startB, SIZE_TYPE lenB){
    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
      
        int iter =0;
        int a_top,b_top,a_bottom,index,a_i,b_i;
    
        //initialisation des variables pour le tri local
        index = i;
        lenA -= startA;
        lenB -= startB;
        A = &A[startA];
        B = &B[startB];
    
        if (index > lenA ){
    
            b_top = index - lenA; //k[0]
            a_top = lenA;     //k[1]  
        }
        else{
            b_top =  0;        //k[0]
            a_top = index;    //k[1]
        }
    
        a_bottom = b_top;      //P[1]
    
    
    
        if( i < (lenA+lenB) ){  //les threads non concerne ne travaillent pas -> sinon loop infini
            while (true) {
                iter++;
    
                int offset = abs(a_top - a_bottom)/2;
    
                a_i = a_top - offset;     //Q[0] = K[0] + offset;
                b_i = b_top + offset;     //Q[1] = K[1] - offset;
    
                if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){
    
                    if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){
    
                        if(a_i < lenA && ( b_i == lenB || A[a_i] <= B[b_i] ) ){
    
                            M[i] = A[a_i];
                            Mvalue[i] = Avalue[a_i];
                        }
                        else{
    
                            M[i] = B[b_i];
                            Mvalue[i] = Bvalue[b_i];
                        }
                        break;
                    }
                    else{
                        a_top = a_i - 1;     // K[1] = b_i - 1;
                        b_top = b_i + 1;     // K[0] = a_i + 1;
                    }
                }
                else{
                    a_bottom = a_i +1;      //P[1] = b_i + 1;
                }
            }
    
        
        }
    }
}
template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__global__
void block_rebalancing_kernel(SIZE_TYPE seg_length, SIZE_TYPE level, KEY_PTR keys[], VALUE_PTR values[],
        SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *unique_update_nodes,
        SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE tree_height, \
        SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane, SIZE_TYPE thres) {

    SIZE_TYPE update_id = blockIdx.x;
    SIZE_TYPE update_node = unique_update_nodes[update_id];
    SIZE_TYPE rebalance_width = seg_length << level;
    SIZE_TYPE linear_idx = -1, idx = -1, lane = -1;
    KEY_PTR key;
    VALUE_PTR value;

    __shared__ KEY_TYPE none_keys[ITEM_PER_THREAD];
    __shared__ VALUE_TYPE none_values[ITEM_PER_THREAD];
    if (threadIdx.x == 0) {
	memset(none_keys, -1, sizeof(KEY_TYPE) * ITEM_PER_THREAD);
	memset(none_values, -1, sizeof(VALUE_TYPE) * ITEM_PER_THREAD);
    }
    __syncthreads();

    if (update_node + rebalance_width >= (seg_length << (tree_height + 1))) {
    	rebalance_width -= seg_length;
	upper_bound -= seg_length;
	if (rebalance_width == 0)
		return;
    }
    if (threadIdx.x * ITEM_PER_THREAD >= rebalance_width) {
	key = none_keys;
	value = none_values;
    }
    else {
    	linear_idx = update_node + threadIdx.x * ITEM_PER_THREAD;
    	idx = GET_IDX(linear_idx, tree_height);
    	lane = GET_LANE(linear_idx, idx, tree_height);
    	key = keys[idx] + lane;
    	value = values[idx] + lane;
    }

    // compact
    __shared__ SIZE_TYPE compacted_size;
    __shared__ KEY_TYPE tmp_keys[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    __shared__ VALUE_TYPE tmp_values[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    block_compact_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(key, value, compacted_size, tmp_keys, tmp_values, thres);
    __syncthreads();

    // judge whether fit the density threshold
    SIZE_TYPE interval_a = update_offset[update_id];
    SIZE_TYPE interval_b = update_offset[update_id + 1];
    SIZE_TYPE interval_size = interval_b - interval_a;
    SIZE_TYPE merge_size = compacted_size + interval_size;
    __syncthreads();


    if (lower_bound <= merge_size && merge_size <= upper_bound) {

        __shared__ KEY_TYPE tmp_keys_sort[THREAD_PER_BLOCK * ITEM_PER_THREAD];
        __shared__ VALUE_TYPE tmp_values_sort[THREAD_PER_BLOCK * ITEM_PER_THREAD];
        SIZE_TYPE THREADNUM = THREAD_PER_BLOCK * ITEM_PER_THREAD;
        //printf("THREADNUM %d \n",THREADNUM);
        mergesmallk<THREAD_PER_BLOCK, ITEM_PER_THREAD>(tmp_keys_sort,tmp_values_sort,merge_size,tmp_keys,tmp_values,0,compacted_size,
            update_keys + interval_a, update_values + interval_a,0,interval_size);
            __syncthreads();
           // printf("pass \n");
	// move
       // block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(tmp_keys + compacted_size, tmp_values + compacted_size,
               // update_keys + interval_a, update_values + interval_a, interval_size);
       // __syncthreads();

        // set SIZE_NONE for executed update
        for (SIZE_TYPE i = interval_a + threadIdx.x; i < interval_b; i += blockDim.x) {
            update_nodes[i] = SIZE_NONE;
        }
        
        // re-dispatch
       //block_redispatch_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(keys, values, rebalance_width, seg_length,
                //merge_size, update_node, tmp_keys, tmp_values, update_node, tree_height, csr_idx, csr_lane);
        block_redispatch_merge_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(keys, values, rebalance_width, seg_length,
                    merge_size, update_node, tmp_keys_sort, tmp_values_sort, update_node, tree_height, csr_idx, csr_lane);

    }
}




__global__
void copy_compacted_kv(SIZE_TYPE *exscan, KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE size, KEY_TYPE *tmp_keys,
        VALUE_TYPE *tmp_values, SIZE_TYPE *compacted_size, SIZE_TYPE update_node, SIZE_TYPE tree_height, SIZE_TYPE thres) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            SIZE_TYPE linear_idx = update_node + i;
            SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
            SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);
            tmp_keys[loc] = keys[idx][lane];
            tmp_values[loc] = values[idx][lane];
        }
    }

    if (0 == global_thread_id) {
        SIZE_TYPE loc = exscan[size - 1];
	   SIZE_TYPE linear_idx = update_node + size - 1;
        SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
        SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);
        if (keys[idx][lane] == KEY_NONE || (keys[idx][lane] & TIME_MASK) < (KEY_TYPE)thres) {
            *compacted_size = loc;
        } else {
            *compacted_size = loc + 1;
            tmp_keys[loc] = keys[idx][lane];
            tmp_values[loc] = values[idx][lane];
        }
    }
}

__global__
void label_key_whether_none_kernel(SIZE_TYPE *label, KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE size, SIZE_TYPE update_node,
    SIZE_TYPE tree_height, SIZE_TYPE thres, SIZE_TYPE fuck=0) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id + update_node; i < size + update_node; i += block_offset) {
    	SIZE_TYPE idx = GET_IDX(i, tree_height);
    	SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
        label[i - update_node] = (keys[idx][lane] == KEY_NONE || (keys[idx][lane] & TIME_MASK) < (KEY_TYPE)thres) ? 0 : 1;
//if(fuck && (keys[idx][lane]>>SRC_SHIFT)==1224622)
//printf("%u,%u,%u,%llu\n",i,idx,lane,(keys[idx][lane]&DST_MASK)>>DST_SHIFT);
    }
}

__device__
void compact_kernel(SIZE_TYPE size, KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE *compacted_size,
        KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *exscan, SIZE_TYPE *label, SIZE_TYPE update_node,
        SIZE_TYPE tree_height, SIZE_TYPE thres) {

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(label, keys, values, size, update_node, tree_height, thres);
    cErr(cudaDeviceSynchronize());

    // exscan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, label, exscan, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, label, exscan, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    // copy compacted kv to tmp, and set the original to none
    copy_compacted_kv<<<BLOCKS_NUM, THREADS_NUM>>>(exscan, keys, values, size, tmp_keys, tmp_values, compacted_size, update_node, tree_height, thres);
    cErr(cudaDeviceSynchronize());
}

__global__
void get_data_kernel(SIZE_TYPE size, KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE *compacted_size,
        KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *exscan, SIZE_TYPE *label, SIZE_TYPE update_node,
        SIZE_TYPE tree_height, SIZE_TYPE thres) {

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(label, keys, values, size, update_node, tree_height, thres);
    cErr(cudaDeviceSynchronize());

    // exscan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, label, exscan, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, label, exscan, size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    // copy compacted kv to tmp, and set the original to none
    copy_compacted_kv<<<BLOCKS_NUM, THREADS_NUM>>>(exscan, keys, values, size, tmp_keys, tmp_values, compacted_size, update_node, tree_height, thres);
    cErr(cudaDeviceSynchronize());

}


__global__
void redispatch_kernel(KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_PTR keys[], VALUE_PTR values[],
        SIZE_TYPE update_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE update_node, SIZE_TYPE tree_height, \
        SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    KEY_TYPE frac = update_width / seg_length;
    KEY_TYPE deno = merge_size;

    for (SIZE_TYPE i = global_thread_id; i < merge_size; i += block_offset) {
        SIZE_TYPE seg_idx = (SIZE_TYPE) ((unsigned long long)frac * (unsigned long long)i / deno);
        SIZE_TYPE seg_lane = (SIZE_TYPE) ((unsigned long long)frac * (unsigned long long)i % deno / frac);
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;
        KEY_TYPE cur_key = tmp_keys[i];
        VALUE_TYPE cur_value = tmp_values[i];
        SIZE_TYPE linear_idx = proj_location + update_node;
        SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
        SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);

        keys[idx][lane] = cur_key;
        values[idx][lane] = cur_value;

        //addition for csr
        if ((cur_key & DST_AND_TIME_MASK) == DST_AND_TIME_END) {
            SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> SRC_SHIFT);
            csr_idx[cur_row + 1] = idx;
            csr_lane[cur_row + 1] = lane;
        }
    }
}


__device__ 
void mergeBig_k(KEY_PTR M,VALUE_PTR Mvalue, KEY_TYPE* A,VALUE_TYPE* Avalue, SIZE_TYPE startA,SIZE_TYPE lenA,KEY_TYPE *B,VALUE_TYPE* Bvalue,SIZE_TYPE startB,SIZE_TYPE lenB, SIZE_TYPE startM){
   
    __shared__ KEY_TYPE s_M[1024];
	__shared__ VALUE_TYPE s_M_value[1024];
    SIZE_TYPE i = threadIdx.x; 
    SIZE_TYPE blockId = blockIdx.x;
    SIZE_TYPE iter =0;
    SIZE_TYPE a_top,b_top,a_bottom,index,a_i,b_i;

    //initialisation des variables pour le tri local
    index = i;
    lenA -= startA;
    lenB -= startB;
    A = &A[startA];
    B = &B[startB];

    if (index > lenA ){

        b_top = index - lenA; //k[0]
        a_top = lenA;     //k[1]  
    }
    else{
        b_top =  0;        //k[0]
        a_top = index;    //k[1]
    }

    a_bottom = b_top;      //P[1]



    if( i < (lenA+lenB) ){  //les threads non concerne ne travaillent pas -> sinon loop infini
        while (true) {
            iter++;

            int offset = fabsf(a_top - a_bottom)/2;

            a_i = a_top - offset;     //Q[0] = K[0] + offset;
            b_i = b_top + offset;     //Q[1] = K[1] - offset;

            if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){

                if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){

                    if(a_i < lenA && ( b_i == lenB || A[a_i] <= B[b_i] ) ){

                        s_M[i] = A[a_i];
						s_M_value[i] = Avalue[a_i];
						//M[startM + i]= A[a_i];
						//Mvalue[startM + i] = Avalue[a_i];
                    }
                    else{

                        s_M[i] = B[b_i];
						s_M_value[i] = Bvalue[b_i];
						//M[startM + i]= B[b_i];
						//Mvalue[startM + i] = Bvalue[b_i];
						
                    }
                    break;
                }
                else{
                    a_top = a_i - 1;     // K[1] = b_i - 1;
                    b_top = b_i + 1;     // K[0] = a_i + 1;
                }
            }
            else{
                a_bottom = a_i +1;      //P[1] = b_i + 1;
            }
        }

    __syncthreads();
    M[startM + i] = s_M[i];
	Mvalue[startM + i] = s_M_value[i];	
    }

}

__global__ 
void pathBig_k(KEY_PTR M,VALUE_PTR Mvalue, KEY_TYPE* A,VALUE_TYPE* Avalue, SIZE_TYPE lenA,KEY_TYPE *B,VALUE_TYPE* Bvalue,SIZE_TYPE lenB){
    //SIZE_TYPE threadId = threadIdx.x;
    SIZE_TYPE i        =  blockIdx.x;
    SIZE_TYPE a_top,b_top,a_bottom,index,a_i,b_i;
  
    SIZE_TYPE A_start; // startA pour chaque block
    SIZE_TYPE B_start; // startB pour chaque block
  
    A_start = lenA;
    B_start = lenB;


    index = i*1024; //indice de l'ement de M par rapport au nlock (initialisation)

    if (index > lenA){

        b_top = index - lenA; //k[0]
        a_top = lenA;     //k[1]  
    }
    else{

        b_top = 0;        //k[0]
        a_top = index;    //k[1]
    }

    a_bottom = b_top;      //P[1]

    //binary search on diag intersections

    while(true){

        SIZE_TYPE offset = fabsf(a_top - a_bottom)/2;

        a_i = a_top - offset;     //Q[0] = K[0] + offset;
        b_i = b_top + offset;     //Q[1] = K[1] - offset;

         if(  (a_i >= 0) && (b_i <= lenB) && ( (a_i == lenA)  || (b_i == 0) || (A[a_i] > B[b_i-1]) ) ){
            //printf("hello\n");

            if( (b_i == lenB) || a_i == 0 || A[a_i-1] <= B[b_i]){
                A_start = a_i;
                B_start = b_i;
                break;
            }
            else{
                a_top = a_i - 1;     // K[1] = b_i - 1;
                b_top = b_i + 1;     // K[0] = a_i + 1;
            }
        }
        else{
            a_bottom = a_i +1;      //P[1] = b_i + 1;
        }
    }

    __syncthreads();
    mergeBig_k(M,Mvalue,A,Avalue,A_start,lenA,B,Bvalue,B_start,lenB,i*1024);
    
}



__global__
void rebalancing_kernel(bool skip, SIZE_TYPE unique_update_size, SIZE_TYPE seg_length, SIZE_TYPE level, KEY_PTR keys[], VALUE_PTR values[],
        SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values,
        SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound,
        SIZE_TYPE tree_height, SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane, SIZE_TYPE thres,
	SIZE_TYPE *compacted_size, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *tmp_exscan,
	SIZE_TYPE *tmp_label, KEY_TYPE *tmp_keys_sorted, VALUE_TYPE *tmp_values_sorted) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    SIZE_TYPE update_width = seg_length << level;

	//get pre allocated tmp memory
	compacted_size += blockIdx.x;
	tmp_keys += (blockIdx.x * update_width);
	tmp_values += (blockIdx.x * update_width);
	tmp_exscan += (blockIdx.x * update_width);
	tmp_label += (blockIdx.x * update_width);
	tmp_keys_sorted += (blockIdx.x * update_width);
	tmp_values_sorted += (blockIdx.x * update_width);

    for (SIZE_TYPE i = global_thread_id; i < unique_update_size; i += block_offset) {
        SIZE_TYPE update_node = unique_update_nodes[i];
    	if (update_node + update_width >= (seg_length << (tree_height + 1))) {
    		update_width -= seg_length;
		upper_bound -= seg_length;
		if (update_width == 0)
			continue;
    	}

        // compact
        compact_kernel(update_width, keys, values, compacted_size, tmp_keys, tmp_values, tmp_exscan, tmp_label, update_node, tree_height, thres);
        cErr(cudaDeviceSynchronize());

        // judge whether fit the density threshold
        SIZE_TYPE interval_a = update_offset[i];
        SIZE_TYPE interval_b = update_offset[i + 1];
        SIZE_TYPE interval_size = interval_b - interval_a;
        SIZE_TYPE merge_size = (*compacted_size) + interval_size;

        if ((lower_bound <= merge_size && merge_size <= upper_bound)&&(!skip)) {
			/*
            SIZE_TYPE THREADS_NUM = 32;
            SIZE_TYPE BLOCKS_NUM;
			BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, interval_size);
            memcpy_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys + (*compacted_size),
                    update_keys + interval_a, interval_size);
            memcpy_kernel<VALUE_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(tmp_values + (*compacted_size),
                    update_values + interval_a, interval_size);
            cErr(cudaDeviceSynchronize());
			
			cub_sort_key_value(tmp_keys, tmp_values, merge_size, tmp_keys_sorted, tmp_values_sorted, update_node);
			*/
			
			SIZE_TYPE THREADS_NUM = 1024;
            SIZE_TYPE BLOCKS_NUM =  (merge_size - 1) / THREADS_NUM + 1;

			//printf("compacted_size= %d, interval_size= %d, blocs= %d\n",(*compacted_size),interval_size,BLOCKS_NUM);
			//SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
			pathBig_k<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys_sorted,tmp_values_sorted, tmp_keys,tmp_values, (*compacted_size),update_keys+ interval_a,update_values+ interval_a,interval_size);
			cErr(cudaDeviceSynchronize());
			//printf("Pass \n");
			
			THREADS_NUM = 32;
			BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, interval_size);
            // set SIZE_NONE for executed updates
            memset_kernel<SIZE_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes + interval_a, SIZE_NONE, interval_size);
            cErr(cudaDeviceSynchronize());
            
           
            // re-dispatch
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
            level_memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(keys, KEY_NONE, update_width, update_node, tree_height);
            cErr(cudaDeviceSynchronize());

            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
            redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys_sorted, tmp_values_sorted, keys, values, update_width, seg_length,
                    merge_size, update_node, tree_height, csr_idx, csr_lane);
            cErr(cudaDeviceSynchronize());

        }

    }

}


__host__
void rebalance_batch(bool skip, SIZE_TYPE level, SIZE_TYPE seg_length, KEY_PTR keys[], VALUE_PTR values[],
        SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
        SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE unique_update_size,
        SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE tree_height, SIZE_TYPE *csr_idx,
	SIZE_TYPE *csr_lane, SIZE_TYPE thres, SIZE_TYPE *compacted_size, KEY_TYPE *tmp_keys,
	VALUE_TYPE *tmp_values, SIZE_TYPE *tmp_exscan, SIZE_TYPE *tmp_label, KEY_TYPE *tmp_keys_sorted,
	VALUE_TYPE *tmp_values_sorted) {

    SIZE_TYPE update_width = seg_length << level;
    if (update_width <= 1024) {
        // func pointer for each template
        void (*func_arr[10])(SIZE_TYPE, SIZE_TYPE, KEY_PTR[], VALUE_PTR[], SIZE_TYPE*, KEY_TYPE*, VALUE_TYPE*,
                SIZE_TYPE*, SIZE_TYPE*, SIZE_TYPE, SIZE_TYPE, SIZE_TYPE, SIZE_TYPE*, SIZE_TYPE*, SIZE_TYPE);
        func_arr[0] = block_rebalancing_kernel<2, 1>;
        func_arr[1] = block_rebalancing_kernel<4, 1>;
        func_arr[2] = block_rebalancing_kernel<8, 1>;
        func_arr[3] = block_rebalancing_kernel<16, 1>;
        func_arr[4] = block_rebalancing_kernel<32, 1>;
        func_arr[5] = block_rebalancing_kernel<32, 2>;
        func_arr[6] = block_rebalancing_kernel<32, 4>;
        func_arr[7] = block_rebalancing_kernel<32, 8>;
        func_arr[8] = block_rebalancing_kernel<32, 16>;
        func_arr[9] = block_rebalancing_kernel<32, 32>;

        // operate each tree node by cuda-block
        SIZE_TYPE THREADS_NUM = update_width > 32 ? 32 : update_width;
        SIZE_TYPE BLOCKS_NUM = unique_update_size;

        func_arr[fls(update_width) - 2]<<<BLOCKS_NUM, THREADS_NUM>>>(seg_length, level, keys, values, update_nodes,
                update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, tree_height, csr_idx, csr_lane, thres);
    } else {
        // operate each tree node by cub-kernel (dynamic parallelsim)
        SIZE_TYPE BLOCKS_NUM = min(16, unique_update_size);
	if (BLOCKS_NUM * (seg_length << level) > MAX_UPDATE_WIDTH) {
		printf("MAX UPDATE WIDTH AT LEAST %lu\n", BLOCKS_NUM * (seg_length << level));
		return;
	}
	//std::cout<<BLOCKS_NUM<<std::endl;
        rebalancing_kernel<<<BLOCKS_NUM, 1>>>(skip, unique_update_size, seg_length, level, keys, values, update_nodes, update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, tree_height, csr_idx, csr_lane, thres, compacted_size, tmp_keys, tmp_values, tmp_exscan, tmp_label, tmp_keys_sorted, tmp_values_sorted);
    }
    cErr(cudaDeviceSynchronize());
}

__global__
void item_counting_kernel(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE *c_sum, SIZE_TYPE size, SIZE_TYPE tree_height, SIZE_TYPE thres) {
	SIZE_TYPE global_idx = blockDim.x * blockIdx.x + threadIdx.x;

	SIZE_TYPE sum = 0;
	for (int i = global_idx; i < size; i += gridDim.x * blockDim.x) {
		SIZE_TYPE idx = GET_IDX(i, tree_height);
		SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
		if (KEY_NONE != keys[idx][lane] && (keys[idx][lane] & TIME_MASK) >= (KEY_TYPE)thres) {
			++sum;
		}
	}

	sum += __shfl_down(sum,16);
	sum += __shfl_down(sum,8);
	sum += __shfl_down(sum,4);
	sum += __shfl_down(sum,2);
	sum += __shfl_down(sum,1);
	if (threadIdx.x % 32 == 0) {
			c_sum[global_idx >> 5] = sum;
	}
	return ;
}

__global__
void level_counting_kernel(KEY_PTR keys, SIZE_TYPE *d_empty_segs, SIZE_TYPE size) {
    SIZE_TYPE global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = global_idx; i < size; i += gridDim.x * blockDim.x) {
        SIZE_TYPE sum = (KEY_NONE != keys[i]);
        sum += __shfl_down(sum,16);
        sum += __shfl_down(sum,8);
        sum += __shfl_down(sum,4);
        sum += __shfl_down(sum,2);
        sum += __shfl_down(sum,1);
        if (threadIdx.x % 32 == 0) {
            if (sum == 0)
                *d_empty_segs = 1;
        }
    }
    return ;
}

__host__
int resize_rpma(RPMA &rpma, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values, SIZE_TYPE update_size, SIZE_TYPE thres, bool reach_root) {

    SIZE_TYPE item_num_include_none = (rpma.segment_length << (rpma.tree_height + 1)) - rpma.segment_length;
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, item_num_include_none);
    SIZE_TYPE SUM_SIZE = THREADS_NUM * BLOCKS_NUM / 32;

	SIZE_TYPE *d_sum, *d_compacted_size, compacted_size;
	SIZE_TYPE updated_tree_height = 0;
	SIZE_TYPE tree_size = rpma.segment_length * ((2 << updated_tree_height) - 1);
	
	cErr(cudaMalloc(&d_sum, sizeof(SIZE_TYPE) * SUM_SIZE));
    cErr(cudaMalloc(&d_compacted_size, sizeof(SIZE_TYPE)));
	cErr(cudaMemset(d_sum, 0, sizeof(SIZE_TYPE) * SUM_SIZE));
	
    item_counting_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), d_sum, item_num_include_none, rpma.tree_height, thres);
    cErr(cudaDeviceSynchronize());

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cErr(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_compacted_size, SUM_SIZE));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_compacted_size, SUM_SIZE));
    cErr(cudaDeviceSynchronize());

    cErr(cudaMemcpy(&compacted_size, d_compacted_size, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));

    cErr(cudaFree(d_temp_storage));
    cErr(cudaFree(d_sum));
    cErr(cudaFree(d_compacted_size));
	cErr(cudaDeviceSynchronize());

    SIZE_TYPE merge_size = compacted_size + update_size;
    SIZE_TYPE original_tree_height = rpma.tree_height;
	//std::cout << "LMerge Size: " <<merge_size<< std::endl;
   
	while (floor(rpma.density_upper_thres_root * tree_size) < merge_size) {
		updated_tree_height += 1;
		tree_size = rpma.segment_length * ((2 << updated_tree_height) - 1);
	}
	if (reach_root) {
		updated_tree_height += 1;
		tree_size = rpma.segment_length * ((2 << updated_tree_height) - 1);
		//return updated_tree_height;
	}
	//if(!rpma.reachroot){
   // assert(original_tree_height != updated_tree_height);}
	//std::cout << "Level Up: " <<updated_tree_height<< std::endl;
    return updated_tree_height;
}


__global__
void up_level_kernel(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
        SIZE_TYPE node = update_nodes[i];
        update_nodes[i] = node & ~update_width;
    }
}

__host__
void up_level_batch(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
    up_level_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes, update_size, update_width);
    cErr(cudaDeviceSynchronize());
}

void show_rpma(RPMA &rpma) {
    for (int i = 0; i <= rpma.tree_height; ++i) {
        SIZE_TYPE size = rpma.segment_length << i;
        KEY_TYPE *h_keys = new KEY_TYPE[size];
        VALUE_TYPE *h_values = new VALUE_TYPE[size];
        cErr(cudaMemcpy(h_keys, rpma.levels_key_ptr_array[i], sizeof(KEY_TYPE) * size, cudaMemcpyDeviceToHost));
        cErr(cudaMemcpy(h_values, rpma.levels_value_ptr_array[i], sizeof(VALUE_TYPE) * size, cudaMemcpyDeviceToHost));
        std::cout << "level " << i << std::endl;	
        std::cout << "\tnum:\tkey:\tvalue:" << std::endl;
        for (int j = 0; j < size; ++j) {
		if (j % rpma.segment_length == 0)
			std::cout << "\tseg " << j / rpma.segment_length << std::endl;
            std::cout << " \t " << j << " \t " << (h_keys[j]>>SRC_SHIFT)<<"  to  " <<((h_keys[j]<<22)>>SRC_SHIFT)<<"  || " << " \t " << h_values[j] << std::endl;
        }
	std::cout << std::endl;
        delete []h_keys;
        delete []h_values;
    }
}

void alloc_or_del_levels(RPMA &rpma, SIZE_TYPE new_height) {
    //assert(rpma.tree_height != new_height);
    
    for (int i = rpma.tree_height + 1; i <= new_height; ++i) {
        KEY_PTR new_level_keys;
        VALUE_PTR new_level_values;
        cErr(cudaMalloc(&new_level_keys, (sizeof(KEY_TYPE) * rpma.segment_length) << i));
        cErr(cudaMalloc(&new_level_values, (sizeof(VALUE_TYPE) * rpma.segment_length) << i));
        cErr(cudaMemset(new_level_keys, -1, (sizeof(KEY_TYPE) * rpma.segment_length) << i));
        rpma.levels_key_ptr_array[i] = new_level_keys;
        rpma.levels_value_ptr_array[i] = new_level_values;
        cErr(cudaDeviceSynchronize());
    }
    for (int i = new_height + 1; i <= rpma.tree_height; ++i) {
        cErr(cudaFree(rpma.levels_key_ptr_array[i]));
        cErr(cudaFree(rpma.levels_value_ptr_array[i]));
        cErr(cudaDeviceSynchronize());
    }
    rpma.tree_height = new_height;
}

void rebuild_rpma(RPMA &rpma, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values, SIZE_TYPE new_height, SIZE_TYPE thres, SIZE_TYPE update_size) {
        SIZE_TYPE old_size = rpma.segment_length * ((2 << rpma.tree_height) - 1);
        SIZE_TYPE new_size = rpma.segment_length * ((2 << new_height) - 1);

        KEY_TYPE *d_keys;
        VALUE_TYPE *d_values;
        SIZE_TYPE *d_exscan;
        SIZE_TYPE *d_label;
        SIZE_TYPE *compacted_size;
        KEY_TYPE *tmp_keys_sorted;
        VALUE_TYPE *tmp_values_sorted;

        cErr(cudaMalloc(&d_keys, new_size * sizeof(KEY_TYPE)));
        cErr(cudaMalloc(&tmp_keys_sorted, new_size * sizeof(KEY_TYPE)));
        cErr(cudaMalloc(&d_values, new_size * sizeof(VALUE_TYPE)));
        cErr(cudaMalloc(&tmp_values_sorted, new_size * sizeof(VALUE_TYPE)));
        cErr(cudaMalloc(&d_exscan, old_size * sizeof(SIZE_TYPE)));
        cErr(cudaMalloc(&d_label, old_size * sizeof(SIZE_TYPE)));
        cErr(cudaMalloc(&compacted_size, sizeof(SIZE_TYPE)));
        cErr(cudaDeviceSynchronize());
		//std::cout << "rebuild" << std::endl;
        get_data_kernel<<<1,1>>>(old_size, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), compacted_size, d_keys, d_values, d_exscan, d_label, 0, rpma.tree_height, thres);
        cErr(cudaDeviceSynchronize());

        SIZE_TYPE h_compacted_size;
        cErr(cudaMemcpy(&h_compacted_size, compacted_size, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
        SIZE_TYPE merge_size = h_compacted_size + update_size;

        /*******************************************************************************************************************************/
        alloc_or_del_levels(rpma, new_height);
        /*******************************************************************************************************************************/

        SIZE_TYPE THREADS_NUM = 32;
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM,SIZE_TYPE(update_size));
        memcpy_kernel<KEY_TYPE> <<<THREADS_NUM, BLOCKS_NUM>>>(d_keys + h_compacted_size, RAW_PTR(update_keys), SIZE_TYPE(update_size));
        cErr(cudaDeviceSynchronize());
        memcpy_kernel<VALUE_TYPE> <<<THREADS_NUM, BLOCKS_NUM>>>(d_values + h_compacted_size, RAW_PTR(update_values), SIZE_TYPE(update_values.size()));
        cErr(cudaDeviceSynchronize());

        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, new_size);
        level_memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(rpma.levels_key_ptr_array), KEY_NONE, new_size, 0, rpma.tree_height);
        cErr(cudaDeviceSynchronize());
    
        global_cub_sort_key_value<<<1,1>>>(d_keys, d_values, merge_size, tmp_keys_sorted, tmp_values_sorted, 0);
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);       
        redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys_sorted, tmp_values_sorted, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), new_size, rpma.segment_length,
                        merge_size, 0, rpma.tree_height, RAW_PTR(rpma.csr_idx), RAW_PTR(rpma.csr_lane));
        
        cErr(cudaFree(d_keys));
        cErr(cudaFree(tmp_keys_sorted));
        cErr(cudaFree(d_values));
        cErr(cudaFree(tmp_values_sorted));
        cErr(cudaFree(d_exscan));
        cErr(cudaFree(d_label));
        cErr(cudaFree(compacted_size));
}

void fill_empty_segs(RPMA &rpma, SIZE_TYPE thres) {
    SIZE_TYPE item_num_include_none = (rpma.segment_length << (rpma.tree_height - 1));
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, item_num_include_none);

    SIZE_TYPE *d_empty_segs;
    SIZE_TYPE empty_segs;
    cErr(cudaMalloc(&d_empty_segs, sizeof(SIZE_TYPE)));
    cErr(cudaMemset(d_empty_segs, 0, sizeof(SIZE_TYPE)));

    //work only for seg_length == 32 
    level_counting_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(rpma.levels_key_ptr_array[rpma.tree_height - 1], d_empty_segs, item_num_include_none);
    cErr(cudaDeviceSynchronize());

    cErr(cudaMemcpy(&empty_segs, d_empty_segs, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    cErr(cudaFree(d_empty_segs));
    cErr(cudaDeviceSynchronize());

        //empty node detected
    if (empty_segs > 0) {
        SIZE_TYPE size = rpma.segment_length * ((2 << rpma.tree_height) - 1);
        KEY_TYPE *d_keys;
        VALUE_TYPE *d_values;
        SIZE_TYPE *d_exscan;
        SIZE_TYPE *d_label;
        SIZE_TYPE *compacted_size;

        cErr(cudaMalloc(&d_keys, size * sizeof(KEY_TYPE)));
        cErr(cudaMalloc(&d_values, size * sizeof(VALUE_TYPE)));
        cErr(cudaMalloc(&d_exscan, size * sizeof(SIZE_TYPE)));
        cErr(cudaMalloc(&d_label, size * sizeof(SIZE_TYPE)));
        cErr(cudaMalloc(&compacted_size, sizeof(SIZE_TYPE)));
        cErr(cudaDeviceSynchronize());
//std::cout << "fill empty segs" << std::endl;
        get_data_kernel<<<1,1>>>(size, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), compacted_size, d_keys, d_values, d_exscan, d_label, 0, rpma.tree_height, thres);
        cErr(cudaDeviceSynchronize());

        SIZE_TYPE h_compacted_size;
        cErr(cudaMemcpy(&h_compacted_size, compacted_size, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));

        SIZE_TYPE THREADS_NUM = 32;

        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
        level_memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(rpma.levels_key_ptr_array), KEY_NONE, size, 0, rpma.tree_height);
        cErr(cudaDeviceSynchronize());
              
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, h_compacted_size);       
        redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(d_keys, d_values, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), size, rpma.segment_length, \
		h_compacted_size, 0, rpma.tree_height, RAW_PTR(rpma.csr_idx), RAW_PTR(rpma.csr_lane));
       
        cErr(cudaFree(d_keys));
        cErr(cudaFree(d_values));
        cErr(cudaFree(d_exscan));
        cErr(cudaFree(d_label));
        cErr(cudaFree(compacted_size));
    }
}

__global__
void check_row_offset_kernel(KEY_PTR *keys, SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane, SIZE_TYPE row_num, SIZE_TYPE tree_height) {
	SIZE_TYPE global_idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = global_idx; i < row_num; i += gridDim.x * blockDim.x) {
		SIZE_TYPE idx = GET_IDX(i, tree_height);
		SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
		KEY_TYPE key = keys[idx][lane];
		SIZE_TYPE src = key >> SRC_SHIFT;
		SIZE_TYPE dst = (key & DST_MASK) >> DST_SHIFT;
		if (src == 1224622)
			printf("%u\n",dst);
	}
}



bool update_rpma(RPMA &rpma, DEV_VEC_KEY &update_keys, DEV_VEC_VALUE &update_values, DEV_VEC_KEY &edgequery, DEV_VEC_VALUE &edgequery_flag, SIZE_TYPE thres) {
	bool levelup=false;
	//int time=0;
	
	//TimeKeeper t1;
	
	
	// step1: sort update keys with values
    thrust::sort_by_key(update_keys.begin(), update_keys.end(), update_values.begin());
    cErr(cudaDeviceSynchronize());
	
    // step2: get leaf node of each update (execute del and mod)
    DEV_VEC_SIZE update_nodes(update_keys.size());
    cErr(cudaDeviceSynchronize());
	//locate_leaf_batch(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), rpma.segment_length, rpma.tree_height,
            		//RAW_PTR(update_keys), RAW_PTR(update_values), update_keys.size(), RAW_PTR(update_nodes));
    locate_leaf_batch_mixed(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), rpma.segment_length, rpma.tree_height,
            RAW_PTR(update_keys), RAW_PTR(update_values), update_keys.size(), RAW_PTR(edgequery), RAW_PTR(edgequery_flag), edgequery.size(),RAW_PTR(update_nodes));
    cErr(cudaDeviceSynchronize());
	
    // step3: extract insertions
    DEV_VEC_SIZE unique_update_nodes(update_keys.size());
    DEV_VEC_SIZE update_offset(update_keys.size() + 1);
    cErr(cudaDeviceSynchronize());
	
    SIZE_TYPE update_size = update_nodes.size();
    SIZE_TYPE unique_node_size = 0;
    compact_insertions(update_nodes, update_keys, update_values, update_size);
	
	/*
	if(rpma.reachroot){
	//TimeKeeper t1;
    SIZE_TYPE original_tree_height = rpma.tree_height;
    SIZE_TYPE updated_tree_height = resize_rpma(rpma, update_keys, update_values, update_size, thres, false);
	//time=t1.checkTime("");
	//std::cout << "resize time,"<< time << std::endl;
	if (updated_tree_height - original_tree_height == 1) {
		std::cout<<"pre level up"<<std::endl;
		levelup=true;
        alloc_or_del_levels(rpma, updated_tree_height);
		fill_empty_segs(rpma, thres);
		locate_leaf_batch(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), rpma.segment_length, rpma.tree_height,
            		RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(update_nodes));
					rpma.reachroot=false;
		}
	}
	*/
    //remove those invalid(KEY_NONE) and replicate insertions
    compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset, unique_node_size);
	
    //construct unique_update_nodes, update_offset and unique_node_size by encode and exc
    cErr(cudaDeviceSynchronize());
	
	//time=t1.checkTime("");
	//std::cout << "search time,"<< time << std::endl;
			
	//check_row_offset_kernel<<<256,256>>>(RAW_PTR(rpma.levels_key_ptr_array), NULL, NULL, (64<<rpma.tree_height)-32, rpma.tree_height);

		// step4: rebalance each tree level
		
	bool skip=false;	
	
    for (SIZE_TYPE level = 0; level <= (rpma.tree_height + 1) && update_size; ++level) {
		//std::cout << "level: " <<level<< std::endl;
			
		if(level==4||level==rpma.tree_height/3){
			skip=true;
		}
		if(level==rpma.tree_height/3-1||level==rpma.tree_height+1){
			skip=false;
		}
		//TimeKeeper t4;
        SIZE_TYPE lower_bound = rpma.lower_element[level];
        SIZE_TYPE upper_bound = rpma.upper_element[level];
		//std::cout << "test 1" << std::endl;
        // re-balance

        rebalance_batch(skip, level, rpma.segment_length, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), RAW_PTR(update_nodes),
                RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(unique_update_nodes),
                RAW_PTR(update_offset), unique_node_size, lower_bound, upper_bound, rpma.tree_height, RAW_PTR(rpma.csr_idx), RAW_PTR(rpma.csr_lane), thres, rpma.compacted_size, rpma.tmp_keys, rpma.tmp_values, rpma.tmp_exscan, rpma.tmp_label, rpma.tmp_keys_sorted, rpma.tmp_values_sorted);
		//std::cout << "test 2" << std::endl;
        // compact
        
            compact_insertions(update_nodes, update_keys, update_values, update_size);
        
		//std::cout << "test 3" << std::endl;
        // up level
        up_level_batch(RAW_PTR(update_nodes), update_size, rpma.segment_length << level);
		//std::cout << "test 4" << std::endl;
		
        // re-compress
        
        compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset,
                unique_node_size);
        
			//std::cout << "test 5" << std::endl;
		
		if((level==rpma.tree_height+1)&& (update_size==0)){
		//if(level==rpma.tree_height){
		    SIZE_TYPE updated_tree_height = resize_rpma(rpma, update_keys, update_values, update_size, thres, true);
			alloc_or_del_levels(rpma, updated_tree_height);
			levelup=true;
			//std::cout<<"pre level up"<<std::endl;
			//rpma.reachroot=true;
		}
			//time=t1.checkTime("");
		//std::cout << "search time,"<< t4.checkTime("") << std::endl;
	
    }
	
//assert(update_keys.size() == update_size);

    //step 5:
    if (update_size > 0) {
		//std::cout<<"#Update Level: "<< update_size<<std::endl;
        SIZE_TYPE original_tree_height = rpma.tree_height;
		
        SIZE_TYPE updated_tree_height = resize_rpma(rpma, update_keys, update_values, update_size, thres, false);
		rpma.reachroot=false;
		//std::cout<<"level up"<<std::endl;
		levelup=true;
    	if (updated_tree_height < original_tree_height || updated_tree_height - original_tree_height > 1) {
		
            rebuild_rpma(rpma, update_keys, update_values, updated_tree_height, thres, update_size);
			
        } else if (updated_tree_height - original_tree_height == 1){
            	alloc_or_del_levels(rpma, updated_tree_height);

    		locate_leaf_batch(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), rpma.segment_length, rpma.tree_height,
            		RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(update_nodes));
		
        	compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset,
                	unique_node_size);

			// step4: rebalance each tree level
			
			
   		for (SIZE_TYPE level = 0; level <= (rpma.tree_height + 1) && update_size; ++level) {
        		SIZE_TYPE lower_bound = rpma.lower_element[level];
        		SIZE_TYPE upper_bound = rpma.upper_element[level];

        		// re-balance
				//std::cout << "b" << (32 << level) << std::endl;
        		rebalance_batch(false,level, rpma.segment_length, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), RAW_PTR(update_nodes), RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(unique_update_nodes), RAW_PTR(update_offset), unique_node_size, lower_bound, upper_bound, rpma.tree_height, RAW_PTR(rpma.csr_idx), RAW_PTR(rpma.csr_lane), thres, rpma.compacted_size, rpma.tmp_keys, rpma.tmp_values, rpma.tmp_exscan, rpma.tmp_label, rpma.tmp_keys_sorted, rpma.tmp_values_sorted);

        		// compact
        		compact_insertions(update_nodes, update_keys, update_values, update_size);

        		// up level
        		up_level_batch(RAW_PTR(update_nodes), update_size, rpma.segment_length << level);

        		// re-compress
       	 		compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset,
                		unique_node_size);
   		 }
		assert(update_size == 0);

            if (rpma.tree_height > 0)
                    fill_empty_segs(rpma, thres);
        } else {
            //should never reach here
            assert(0);
        }
    }
    //assert(update_size == 0);
    cErr(cudaDeviceSynchronize());
    return levelup;
//std::cout << 3 << std::endl;
//check_row_offset_kernel<<<256,256>>>(RAW_PTR(rpma.levels_key_ptr_array), NULL, NULL, (64<<rpma.tree_height)-32, rpma.tree_height);
}

void destroy_rpma(RPMA &rpma) {
	for (int i = 0; i <= rpma.tree_height; ++i) {
		cErr(cudaFree(rpma.levels_key_ptr_array[i]));
		cErr(cudaFree(rpma.levels_value_ptr_array[i]));
	}
    cErr(cudaFree(rpma.compacted_size));
    cErr(cudaFree(rpma.tmp_keys));
    cErr(cudaFree(rpma.tmp_values));
    cErr(cudaFree(rpma.tmp_exscan));
    cErr(cudaFree(rpma.tmp_label));
    cErr(cudaFree(rpma.tmp_keys_sorted));
    cErr(cudaFree(rpma.tmp_values_sorted));
}


SIZE_TYPE get_data_rpma(RPMA &rpma, KEY_PTR &h_keys, VALUE_PTR &h_values) {
    	SIZE_TYPE size = rpma.segment_length * ((2 << rpma.tree_height) - 1);

	DEV_VEC_KEY d_keys(size);
	DEV_VEC_VALUE d_values(size);
	DEV_VEC_SIZE d_exscan(size);
	DEV_VEC_SIZE d_label(size);
	SIZE_TYPE *compacted_size;
	cErr(cudaMalloc(&compacted_size, sizeof(SIZE_TYPE)));
    	cErr(cudaDeviceSynchronize());

	get_data_kernel<<<1,1>>>(size, RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array), compacted_size, \
		RAW_PTR(d_keys), RAW_PTR(d_values), RAW_PTR(d_exscan), RAW_PTR(d_label), 0, rpma.tree_height, 0);
    	cErr(cudaDeviceSynchronize());

	SIZE_TYPE h_compacted_size;
	cErr(cudaMemcpy(&h_compacted_size, compacted_size, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));

	h_keys = new KEY_TYPE[h_compacted_size];
	h_values = new VALUE_TYPE[h_compacted_size];
	cErr(cudaMemcpy(h_keys, RAW_PTR(d_keys), sizeof(KEY_TYPE) * h_compacted_size, cudaMemcpyDeviceToHost));
	cErr(cudaMemcpy(h_values, RAW_PTR(d_values), sizeof(VALUE_TYPE) * h_compacted_size, cudaMemcpyDeviceToHost));

	cErr(cudaFree(compacted_size));

	return h_compacted_size;
}

__host__
void init_rpma(RPMA &rpma) {

    rpma.levels_key_ptr_array.resize(26);
    rpma.levels_value_ptr_array.resize(26);
    rpma.tree_height = 0;
    rpma.segment_length = 32;

    KEY_PTR level_0_key;
    VALUE_PTR level_0_value;
    cErr(cudaMalloc(&level_0_key, sizeof(KEY_TYPE) * rpma.segment_length));
    cErr(cudaMalloc(&level_0_value, sizeof(VALUE_TYPE) * rpma.segment_length));
    cErr(cudaMemset(level_0_key, -1, sizeof(KEY_TYPE) * rpma.segment_length));
    
    //KEY_TYPE k_max = KEY_MAX;
    //cErr(cudaMemcpy(level_0_key, &k_max, sizeof(KEY_TYPE), cudaMemcpyHostToDevice));

    rpma.levels_key_ptr_array[0] = level_0_key;
    rpma.levels_value_ptr_array[0] = level_0_value;

    recalculate_density(rpma);

	//pre allocate structures for rebalancing_kernel
    cErr(cudaMalloc(&rpma.compacted_size, sizeof(SIZE_TYPE) * 64));
    cErr(cudaMalloc(&rpma.tmp_keys, MAX_UPDATE_WIDTH * sizeof(KEY_TYPE)));
    cErr(cudaMalloc(&rpma.tmp_values, MAX_UPDATE_WIDTH * sizeof(VALUE_TYPE)));
    cErr(cudaMalloc(&rpma.tmp_exscan, MAX_UPDATE_WIDTH * sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&rpma.tmp_label, MAX_UPDATE_WIDTH * sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&rpma.tmp_keys_sorted, MAX_UPDATE_WIDTH * sizeof(KEY_TYPE)));
    cErr(cudaMalloc(&rpma.tmp_values_sorted, MAX_UPDATE_WIDTH * sizeof(VALUE_TYPE)));
    cErr(cudaDeviceSynchronize());
	
}

template<typename T>
struct col_idx_none {
    typedef T argument_type;
    typedef T result_type;
    __host__ __device__
    T operator()(const T &x) const {
        return (x << SRC_SHIFT) + DST_AND_TIME_END;
    }
};

__host__
void init_csr_rpma(RPMA &rpma, SIZE_TYPE row_num) {
    rpma.row_num = row_num;
    rpma.csr_idx.resize(row_num + 1, 0);
    rpma.csr_lane.resize(row_num + 1, 0);

    DEV_VEC_KEY row_wall(row_num);
    DEV_VEC_VALUE tmp_value(row_num, 0);
	DEV_VEC_KEY edgequery(row_num);
    DEV_VEC_VALUE edgequery_flag(row_num, 0);
    cErr(cudaDeviceSynchronize());

    thrust::tabulate(row_wall.begin(), row_wall.end(), col_idx_none<KEY_TYPE>());
    init_rpma(rpma);
    cErr(cudaDeviceSynchronize());
	
    update_rpma(rpma, row_wall, tmp_value,edgequery,edgequery_flag, 0);  
}


__global__
void get_nbrlist_len_and_offset_kernel(SIZE_TYPE *query_srcs, SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane, \
    SIZE_TYPE* nbrlist_len, SIZE_TYPE *offset, SIZE_TYPE size, SIZE_TYPE height) {

    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = global_idx; i < size; i += (gridDim.x * blockDim.x)) {
        SIZE_TYPE node = query_srcs[i];
        SIZE_TYPE linear_idx1 = node == 0 ? 0 : (GET_LINEAR_IDX(csr_idx[node], csr_lane[node], height) + (csr_lane[node] % 32));
        SIZE_TYPE linear_idx2 = GET_LINEAR_IDX(csr_idx[node + 1], csr_lane[node + 1], height) + (csr_lane[node + 1] % 32);
        nbrlist_len[i] = linear_idx2 - linear_idx1;
    }

    cudaDeviceSynchronize();

    if (global_idx == 0) {
    	void *d_temp_storage = NULL;
    	size_t temp_storage_bytes = 0;
    	cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nbrlist_len, offset, size + 1));
    	cErr(cudaDeviceSynchronize());
    	cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    	cErr(cudaDeviceSynchronize());
    	cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nbrlist_len, offset, size + 1));
    	cErr(cudaDeviceSynchronize());
    	cErr(cudaFree(d_temp_storage));
    }
}

__global__
void get_nbrlist_kernel(KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE *query_srcs, SIZE_TYPE *nbr_list, SIZE_TYPE *nbrlist_offset, \
    SIZE_TYPE *nbrlist_len, SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane, SIZE_TYPE srcs_num, SIZE_TYPE total_nbrlist_len, SIZE_TYPE tree_height, \
    KEY_TYPE *constraint, VALUE_PTR v_list) {

    SIZE_TYPE global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (SIZE_TYPE i = global_idx; i < total_nbrlist_len; i += (gridDim.x * blockDim.x)) {

        //binary search which node the i-th global edge belongs to
        SIZE_TYPE l = 0, r = srcs_num - 1, mid;
        while (l < r) {
            mid = (l + r + 1) / 2;
            SIZE_TYPE mid_offset = nbrlist_offset[mid];
            if (i > mid_offset)
                l = mid;
            else if (i < mid_offset)
                r = mid - 1;
            else
                break;
        }
        //this condition is impossible
        //while (nbrlist_offset[tmp] == nbrlist_offset[tmp + 1])
           // ++tmp;

        SIZE_TYPE node = (l == r ? l : mid);
        SIZE_TYPE src = query_srcs[node];
        SIZE_TYPE lb = constraint[node] >> 32;
        SIZE_TYPE ub = constraint[node] << 32 >> 32;
        SIZE_TYPE nbr_num = i - nbrlist_offset[node];

        SIZE_TYPE linear_idx = (src == 0 ? 0 : (GET_LINEAR_IDX(csr_idx[src], csr_lane[src], tree_height)) + (csr_lane[src] % 32)) + nbr_num;
        SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
        SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);

        KEY_TYPE nbr_and_time = keys[idx][lane] & DST_AND_TIME_MASK;
        SIZE_TYPE nbr = (nbr_and_time & DST_MASK) >> DST_SHIFT;
        SIZE_TYPE time = nbr_and_time & TIME_MASK;

        if (nbr == NBR_END || (/*values[idx][lane] != VALUE_NONE &&*/time >= lb && time <= ub)) {
        	nbr_list[i] = nbr;
		v_list[i] = values[idx][lane];
	}
        else
        	nbr_list[i] = SIZE_NONE;
    }
}

__global__
void copy_compacted_nbrlist_kernel(SIZE_TYPE *exscan, SIZE_TYPE *tmp_nbrlist, SIZE_TYPE *nbr_list, \
    SIZE_TYPE *compacted_size, SIZE_TYPE size, VALUE_TYPE *tmp_vlist, VALUE_TYPE *v_list) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            tmp_nbrlist[loc] = nbr_list[i];
	    tmp_vlist[loc] = v_list[i];
        }
    }
    if (0 == global_thread_id) {
        SIZE_TYPE loc = exscan[size - 1];
        if (nbr_list[size - 1] == SIZE_NONE) {
            *compacted_size = loc;
        } else {
            *compacted_size = loc + 1;
            tmp_nbrlist[loc] = nbr_list[size - 1];
	    tmp_vlist[loc] = v_list[size - 1];
        }
    }
}

__global__
void label_edge_whether_none_kernel(SIZE_TYPE *nbr_list, SIZE_TYPE *label, SIZE_TYPE size) {
    SIZE_TYPE global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (SIZE_TYPE i = global_idx; i < size; i += (gridDim.x * blockDim.x)) {
        label[i] = (nbr_list[i] == SIZE_NONE ? 0 : 1);
    }
}

__host__
SIZE_TYPE compact_nbrlist(SIZE_TYPE *nbr_list, SIZE_TYPE *nbrlist_offset, SIZE_TYPE *nbrlist_len, \
    SIZE_TYPE *&tmp_nbrlist, SIZE_TYPE total_nbrlist_len, VALUE_TYPE *v_list, VALUE_TYPE *&tmp_vlist) {

    DEV_VEC_SIZE label(total_nbrlist_len), exscan(total_nbrlist_len);

    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, total_nbrlist_len);
    label_edge_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(nbr_list, RAW_PTR(label), total_nbrlist_len);
    cErr(cudaDeviceSynchronize());

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, RAW_PTR(label), RAW_PTR(exscan), total_nbrlist_len));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, RAW_PTR(label), RAW_PTR(exscan), total_nbrlist_len));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    SIZE_TYPE *compacted_size, h_compacted_size;
    cErr(cudaMalloc(&compacted_size, sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&tmp_nbrlist, sizeof(SIZE_TYPE) * total_nbrlist_len));
    cErr(cudaMalloc(&tmp_vlist, sizeof(VALUE_TYPE) * total_nbrlist_len));
    cErr(cudaDeviceSynchronize());
    copy_compacted_nbrlist_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(exscan), tmp_nbrlist, nbr_list, compacted_size, total_nbrlist_len, \
tmp_vlist, v_list);
    cErr(cudaDeviceSynchronize());

    cErr(cudaMemcpy(&h_compacted_size, compacted_size, sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    cErr(cudaFree(compacted_size));

    return h_compacted_size;
}



__host__
void query_rpma(RPMA &rpma, DEV_VEC_SIZE &d_query_srcs, DEV_VEC_KEY &d_query_constraint, \
thrust::host_vector<SIZE_TYPE> &offset, thrust::host_vector<SIZE_TYPE> &len, thrust::host_vector<SIZE_TYPE> &edgelist, thrust::host_vector<VALUE_TYPE> &v_list) {

    SIZE_TYPE srcs_num = d_query_srcs.size();
    thrust::sort_by_key(d_query_srcs.begin(), d_query_srcs.end(), d_query_constraint.begin());
	

    DEV_VEC_SIZE d_nbrlist_len(srcs_num + 2), nbrlist_offset(srcs_num + 2);
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, srcs_num);
    get_nbrlist_len_and_offset_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(d_query_srcs), RAW_PTR(rpma.csr_idx), RAW_PTR(rpma.csr_lane), \
        RAW_PTR(d_nbrlist_len), RAW_PTR(nbrlist_offset), srcs_num, rpma.tree_height);
    cErr(cudaDeviceSynchronize());


    SIZE_TYPE total_nbrlist_len = nbrlist_offset[srcs_num] + 1;
    DEV_VEC_SIZE nbr_list(total_nbrlist_len);
    DEV_VEC_VALUE nbr_vlist(total_nbrlist_len);
    cErr(cudaDeviceSynchronize());


    THREADS_NUM = 32;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, total_nbrlist_len);
    get_nbrlist_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.levels_value_ptr_array),\
        RAW_PTR(d_query_srcs), RAW_PTR(nbr_list), RAW_PTR(nbrlist_offset), RAW_PTR(d_nbrlist_len), RAW_PTR(rpma.csr_idx), RAW_PTR(rpma.csr_lane),\
        srcs_num, total_nbrlist_len, rpma.tree_height, RAW_PTR(d_query_constraint), RAW_PTR(nbr_vlist));
    cErr(cudaDeviceSynchronize());


    //now filter out NONE nbrs in nbrlist
    SIZE_TYPE *new_nbr_list;
    VALUE_TYPE *new_v_list;
    SIZE_TYPE compacted_size = compact_nbrlist(RAW_PTR(nbr_list), RAW_PTR(nbrlist_offset), \
        RAW_PTR(d_nbrlist_len), new_nbr_list, total_nbrlist_len, RAW_PTR(nbr_vlist), new_v_list);
	cErr(cudaDeviceSynchronize());

    edgelist.resize(compacted_size);
    v_list.resize(compacted_size);
    cErr(cudaMemcpy(RAW_PTR(edgelist), new_nbr_list, sizeof(SIZE_TYPE) * compacted_size, cudaMemcpyDeviceToHost));
    cErr(cudaMemcpy(RAW_PTR(v_list), new_v_list, sizeof(VALUE_TYPE) * compacted_size, cudaMemcpyDeviceToHost));
    cErr(cudaFree(new_nbr_list));
    cErr(cudaFree(new_v_list));

    //for now just do it on cpu
    offset.resize(srcs_num);
    len.resize(srcs_num);
	
    offset[0] = (d_query_srcs[0] != 0);
    SIZE_TYPE i, j;
    for (i = 0, j = (d_query_srcs[0] != 0); i < srcs_num && j < compacted_size; ++j) {
        if (edgelist[j] == NBR_END) {
            offset[i + 1] = j + 1;
            len[i] = j - offset[i];
            ++i;
        }
    }
	//std::cout<< "Src: "<< srcs_num <<" size " <<compacted_size<< std::endl;
    assert(i == srcs_num && j == compacted_size);

}



__global__
void check_edge_kernel(KEY_PTR keys[], SIZE_TYPE *csr_idx, SIZE_TYPE *csr_lane, KEY_PTR edges, SIZE_TYPE *res, \
    SIZE_TYPE edge_num, SIZE_TYPE height) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = global_idx; i < edge_num; i += (gridDim.x * blockDim.x)) {
        KEY_TYPE edge = edges[i];
        SIZE_TYPE src = edge >> SRC_SHIFT;

        SIZE_TYPE idx_bg = src == 0 ? (GET_IDX(0, height)) : csr_idx[src];
        SIZE_TYPE idx_ed = csr_idx[src + 1];
        SIZE_TYPE lane_bg = src == 0 ? (GET_LANE(0, idx_bg, height)) : csr_lane[src];
        SIZE_TYPE lane_ed = csr_lane[src + 1];

        while(!(idx_bg == idx_ed && lane_bg == lane_ed)) {
            KEY_TYPE key = keys[idx_bg][lane_bg];
            if ((key & ~TIME_MASK) == (edge & ~TIME_MASK))
                res[i] = 1;
            if ((lane_bg % 32) != 31)
                ++lane_bg;
            else {
                SIZE_TYPE linear_idx = GET_LINEAR_IDX(idx_bg, lane_bg, height) + 32;
                idx_bg = GET_IDX(linear_idx, height);
                lane_bg = GET_LANE(linear_idx, idx_bg, height);
            }
        }
    }
}

/*
__host__
void edge_query_rpma(RPMA &rpma, DEV_VEC_KEY &edges, thrust::host_vector<SIZE_TYPE> &res) {

    SIZE_TYPE edge_num = edges.size();

    thrust::sort(edges.begin(), edges.end());

    DEV_VEC_SIZE d_res(edge_num, 0);
    res.resize(edge_num, 0);

    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, edge_num);
    check_edge_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(rpma.levels_key_ptr_array), RAW_PTR(rpma.csr_idx), \
        RAW_PTR(rpma.csr_lane), RAW_PTR(edges), RAW_PTR(d_res), edge_num, rpma.tree_height);
    cErr(cudaDeviceSynchronize());

    cErr(cudaMemcpy(RAW_PTR(res), RAW_PTR(d_res), sizeof(SIZE_TYPE) * edge_num, cudaMemcpyDeviceToHost));
}
*/


__global__
void edge_query_kernel(SIZE_TYPE *res, KEY_PTR key_ptrs[], SIZE_TYPE seg_length,
	SIZE_TYPE tree_height, KEY_TYPE *update_keys, SIZE_TYPE update_size) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    __shared__ KEY_PTR block_key_ptrs[26];
    //__shared__ VALUE_PTR block_value_ptrs[26];
    __shared__ KEY_TYPE levels_key[32 * ((1<<5)-1)];	
    //__shared__ VALUE_TYPE levels_value[32 * ((1<<5)-1)];	

	if (threadIdx.x == 0) {
		int thres = (tree_height > 4 ? 4 : tree_height);
		for (int i = 0; i <= thres; ++i) {
			int offset = (32 << i) - 32;
			memcpy(levels_key + offset, key_ptrs[i], sizeof(KEY_TYPE) * (32 << i));
			//memcpy(levels_value + offset, value_ptrs[i], sizeof(VALUE_TYPE) * (32 << i));
			block_key_ptrs[i] = levels_key + offset;
			//block_value_ptrs[i] = levels_value + offset;
		}
		memcpy(block_key_ptrs + thres + 1, key_ptrs + thres + 1, sizeof(KEY_PTR) * (25 - thres));
		//memcpy(block_value_ptrs + thres + 1, value_ptrs + thres + 1, sizeof(VALUE_PTR) * (25 - thres));
	}

    __syncthreads();

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
 
        KEY_TYPE key = update_keys[i];
        //VALUE_TYPE value = update_values[i];
    	SIZE_TYPE idx = 0, lane = 0;
    	while(idx < tree_height && KEY_NONE != block_key_ptrs[idx][lane]) {

    		if (block_key_ptrs[idx][lane] > key) {
    			++idx;
    			lane <<= 1;
    			continue;
    		}
    		
    		SIZE_TYPE l = 0, r = seg_length - 1;
    		while(l < r) {
    			SIZE_TYPE mid  = (l + r + 1) >> 1;
    			if (block_key_ptrs[idx][lane + mid] == KEY_NONE)
    				r = mid - 1;
    			else
    				l = mid;
    		}			
    		
    		if (key > block_key_ptrs[idx][lane + l]) {
    			++idx;
                    	lane = (lane << 1) + seg_length;
    		}
            else if(key == block_key_ptrs[idx][lane + l]) {
                res[i] = 1;
            }
    			break;			

    	}
    	//SIZE_TYPE prefix = (seg_length << (tree_height - idx)) - seg_length + (lane << (tree_height - idx + 1));
        //SIZE_TYPE prefix = GET_LINEAR_IDX(idx, lane, tree_height);
        //prefix = handle_del_mod(block_key_ptrs, block_value_ptrs, seg_length, key, value, prefix, idx, lane);
        //leaf[i] = prefix;
    }
}

__host__
void edge_query_batch(SIZE_TYPE *res, KEY_PTR keys[], SIZE_TYPE seg_length,
        SIZE_TYPE tree_height, KEY_TYPE *update_keys, SIZE_TYPE update_size) {

    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);

    edge_query_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(res, keys, seg_length, tree_height, update_keys, update_size);
    cErr(cudaDeviceSynchronize());
}

bool edge_query_rpma(thrust::host_vector<SIZE_TYPE> &res, RPMA &rpma, DEV_VEC_KEY &update_keys, SIZE_TYPE thres) {
	
    SIZE_TYPE edge_num = update_keys.size();

    //thrust::sort(update_keys.begin(), update_keys.end());

    DEV_VEC_SIZE d_res(edge_num, 0);
    
    res.resize(edge_num, 0);
    cErr(cudaDeviceSynchronize());
	
	
    // step2: get leaf node of each update (execute del and mod)
    //DEV_VEC_SIZE update_nodes(update_keys.size());
    //cErr(cudaDeviceSynchronize());
	
	
    edge_query_batch(RAW_PTR(d_res), RAW_PTR(rpma.levels_key_ptr_array), rpma.segment_length, rpma.tree_height,
            RAW_PTR(update_keys), update_keys.size());
    cErr(cudaDeviceSynchronize());
    cErr(cudaMemcpy(RAW_PTR(res), RAW_PTR(d_res), sizeof(SIZE_TYPE) * edge_num, cudaMemcpyDeviceToHost));
	//std::cout<< " Edge size " <<res.size()<< std::endl;

}

__global__
void copy_compacted_col_idx_kernel(SIZE_TYPE *exscan, KEY_PTR keys[], VALUE_PTR values[], SIZE_TYPE size, \
    SIZE_TYPE *col_index, VALUE_TYPE *tmp_values, SIZE_TYPE tree_height) {

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            SIZE_TYPE idx = GET_IDX(i, tree_height);
            SIZE_TYPE lane = GET_LANE(i, idx, tree_height);
            SIZE_TYPE nbr = (keys[idx][lane] & DST_MASK) >> DST_SHIFT;
            if (nbr == NBR_END) {
                col_index[loc] = (keys[idx][lane] >> SRC_SHIFT);
                tmp_values[loc] = VALUE_NONE;
            } else {
                col_index[loc] = nbr;
                tmp_values[loc] = values[idx][lane];
            }
        }
    }

    if (0 == global_thread_id) {
        SIZE_TYPE loc = exscan[size - 1];
        SIZE_TYPE linear_idx = size - 1;
        SIZE_TYPE idx = GET_IDX(linear_idx, tree_height);
        SIZE_TYPE lane = GET_LANE(linear_idx, idx, tree_height);
        if (keys[idx][lane] == KEY_NONE) {
            ;
        } else {
            SIZE_TYPE nbr = (keys[idx][lane] & DST_MASK) >> DST_SHIFT;
            if (nbr == NBR_END) {
                col_index[loc] = (keys[idx][lane] >> SRC_SHIFT);
                tmp_values[loc] = VALUE_NONE;
            } else {
                col_index[loc] = nbr;
                tmp_values[loc] = values[idx][lane];
            }
        }
    }
}


__global__
void construct_row_offset_kernel(SIZE_TYPE *row_offset, SIZE_TYPE *col_index, VALUE_TYPE *values, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        if (values[i] == VALUE_NONE) {
            SIZE_TYPE src = col_index[i];
            row_offset[src + 1] = i + 1;
        }
    }
}

__host__
void to_csr_rpma(RPMA &rpma, DEV_VEC_SIZE &row_offset, DEV_VEC_SIZE &col_index) {

    SIZE_TYPE size = (64 << rpma.tree_height) - 32;

    DEV_VEC_SIZE exscan(size);
    DEV_VEC_SIZE label(size);
    cErr(cudaDeviceSynchronize());

    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(label), RAW_PTR(rpma.levels_key_ptr_array), \
        RAW_PTR(rpma.levels_value_ptr_array), size, 0, rpma.tree_height, 0);
    cErr(cudaDeviceSynchronize());

    // exscan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, RAW_PTR(label), RAW_PTR(exscan), size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cudaDeviceSynchronize());
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, RAW_PTR(label), RAW_PTR(exscan), size));
    cErr(cudaDeviceSynchronize());
    cErr(cudaFree(d_temp_storage));

    SIZE_TYPE compacted_size = exscan[size - 1] + label[size - 1];

    col_index.resize(compacted_size);
    DEV_VEC_VALUE values(compacted_size);

    THREADS_NUM = 32;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    copy_compacted_col_idx_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(exscan), RAW_PTR(rpma.levels_key_ptr_array), \
        RAW_PTR(rpma.levels_value_ptr_array), size, RAW_PTR(col_index), RAW_PTR(values), rpma.tree_height);
    cErr(cudaDeviceSynchronize());

    row_offset.resize(rpma.row_num + 1);
    row_offset[0] = 0;
    row_offset[rpma.row_num] = compacted_size;

    THREADS_NUM = 32;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, compacted_size);
    construct_row_offset_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(row_offset), RAW_PTR(col_index), RAW_PTR(values), \
        compacted_size);
    cErr(cudaDeviceSynchronize());
}

