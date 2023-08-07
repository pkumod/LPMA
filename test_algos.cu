#include <iostream>
#include "util.cuh"
#include <nvgraph.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <cassert>
using namespace std;

void
read_edgelist(string &filename, vector<vector<int> > &G, size_t &node_num, size_t &edge_num, size_t n) {
	ifstream e_file;
	e_file.open((filename + ".edgelist").c_str());

	e_file >> node_num;
	e_file >> edge_num;
	if (n)
		edge_num = n;

	G.resize(node_num);
	for (int i = 0; i < edge_num; ++i) {
		size_t src, dst;
		e_file >> src;
		e_file >> dst;

		assert(src < node_num && dst < node_num);
		G[src].push_back(dst);
	}
	for (int i = 0; i < node_num; ++i) {
		sort(G[i].begin(), G[i].end());
	}

	e_file.close();
}

void check_status(nvgraphStatus_t status){
	    if ((int)status != 0)    {
		            printf("ERROR : %d\n",status);
			            exit(0);
				        }
}


int pgrank(size_t n, size_t nnz, int *destination_offsets_h, int *source_indices_h) {
	    size_t vert_sets = 2, edge_sets = 1;
	        float alpha1 = 0.9f; void *alpha1_p = (void *) &alpha1;
		    // nvgraph variables
		    nvgraphHandle_t handle; nvgraphGraphDescr_t graph;
		        nvgraphCSCTopology32I_t CSC_input;
			    cudaDataType_t edge_dimT = CUDA_R_32F;
			        cudaDataType_t* vertex_dimT;
				    // Allocate host data
				    float *pr_1 = (float*)malloc(n*sizeof(float));
				        void **vertex_dim = (void**)malloc(vert_sets*sizeof(void*));
					    vertex_dimT = (cudaDataType_t*)malloc(vert_sets*sizeof(cudaDataType_t));
					        CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
						    // Initialize host data
						    float *weights_h = new float[n];
							        float *bookmark_h = new float[n];
								for (int i = 0; i < n; ++i) {
									weights_h[i] = 1.f / n;
									bookmark_h[i] = 0.f;
								}
								bookmark_h[1] = 1.f;
								    vertex_dim[0] = (void*)bookmark_h; vertex_dim[1]= (void*)pr_1;
								        vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F, vertex_dimT[2]= CUDA_R_32F;
									    // Starting nvgraph
									    check_status(nvgraphCreate (&handle));
									        check_status(nvgraphCreateGraphDescr (handle, &graph));
										    CSC_input->nvertices = n; CSC_input->nedges = nnz;
										        CSC_input->destination_offsets = destination_offsets_h;
											    CSC_input->source_indices = source_indices_h;
											        // Set graph connectivity and properties (tranfers)
											        check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
												    check_status(nvgraphAllocateVertexData(handle, graph, vert_sets, vertex_dimT));
												        check_status(nvgraphAllocateEdgeData  (handle, graph, edge_sets, &edge_dimT));
													    for (int i = 0; i < 2; ++i)
														            check_status(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
													        check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
											TimeKeeper tk;			    
														    check_status(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));
											cout <<"Pagerank: " << tk.checkTime("") << endl;	
											    			    // Get result
														        check_status(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
															    check_status(nvgraphDestroyGraphDescr(handle, graph));
															        check_status(nvgraphDestroy(handle));
																    free(pr_1); free(vertex_dim); free(vertex_dimT);
																        free(CSC_input);
																	    return 0;
}



int tc(size_t n, size_t nnz, int *source_offsets, int *destination_indices) 
{
	    // nvgraph variables
	    nvgraphHandle_t handle;
	        nvgraphGraphDescr_t graph;
		    nvgraphCSRTopology32I_t CSR_input;

		        // Init host data
		        CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

			    // Undirected graph:
			    // 0       2-------4       
			    //  \     / \     / \
			    //   \   /   \   /   \
			    //    \ /     \ /     \
			    //     1-------3-------5
			    // 3 triangles
			    // CSR of lower triangular of adjacency matrix:

				        check_status(nvgraphCreate(&handle));
					    check_status(nvgraphCreateGraphDescr (handle, &graph));
					        CSR_input->nvertices = n; 
						    CSR_input->nedges = nnz;
						        CSR_input->source_offsets = source_offsets;
							    CSR_input->destination_indices = destination_indices;
							        // Set graph connectivity
							        check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));

								    uint64_t trcount = 0;
	TimeKeeper tk;
								        check_status(nvgraphTriangleCount(handle, graph, &trcount));
	cout <<"TC: " << trcount<<", "<< tk.checkTime("") << endl;    

									        free(CSR_input);
										    check_status(nvgraphDestroyGraphDescr(handle, graph));
										        check_status(nvgraphDestroy(handle));
											    return 0;
}




int bfs(size_t n, size_t nnz, int *source_offsets_h, int *destination_indices_h){
	    //Example of graph (CSR format)
	    const size_t  vertex_numsets = 2, edge_numset = 0;
		        //where to store results (distances from source) and where to store results (predecessors in search tree) 
		        int *bfs_distances_h = new int[n], *bfs_predecessors_h = new int[n];
			    // nvgraph variables
			    nvgraphStatus_t status;
			        nvgraphHandle_t handle;
				    nvgraphGraphDescr_t graph;
				        nvgraphCSRTopology32I_t CSR_input;
					    cudaDataType_t* vertex_dimT;
					        size_t distances_index = 0;
						    size_t predecessors_index = 1;
						        vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
							    vertex_dimT[distances_index] = CUDA_R_32I;
							        vertex_dimT[predecessors_index] = CUDA_R_32I;
								    //Creating nvgraph objects
								    check_status(nvgraphCreate (&handle));
								        check_status(nvgraphCreateGraphDescr (handle, &graph));
									    // Set graph connectivity and properties (tranfers)
									    CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
									        CSR_input->nvertices = n;
										    CSR_input->nedges = nnz;
										        CSR_input->source_offsets = source_offsets_h;
											    CSR_input->destination_indices = destination_indices_h;
											        check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));
												    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
												        int source_vert = 0;
													    //Setting the traversal parameters  
													    nvgraphTraversalParameter_t traversal_param;
													        nvgraphTraversalParameterInit(&traversal_param);
														    nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
														        nvgraphTraversalSetPredecessorsIndex(&traversal_param, predecessors_index);
															    nvgraphTraversalSetUndirectedFlag(&traversal_param, false);
															        //Computing traversal using BFS algorithm
	TimeKeeper tk;
															        check_status(nvgraphTraversal(handle, graph, NVGRAPH_TRAVERSAL_BFS, &source_vert, traversal_param));
	cout << tk.checkTime("") << endl;    
																// Get result
																    check_status(nvgraphGetVertexData(handle, graph, (void*)bfs_distances_h, distances_index));
																        check_status(nvgraphGetVertexData(handle, graph, (void*)bfs_predecessors_h, predecessors_index));
																	    // expect bfs distances_h = (1 0 1 3 3 2 2147483647)
				//													    for (int i = 0; i<n; i++)  printf("Distance to vertex %d: %i\n",i, bfs_distances_h[i]); printf("\n");
																	        // expect bfs predecessors = (1 -1 1 5 5 0 -1)
				//													        for (int i = 0; i<n; i++)  printf("Predecessor of vertex %d: %i\n",i, bfs_predecessors_h[i]); printf("\n");
																		    free(vertex_dimT);
																		        free(CSR_input);
																			    check_status(nvgraphDestroyGraphDescr (handle, graph));
																			        check_status(nvgraphDestroy (handle));
																				    return 0;
}


void test(string dataset, int edge_n) {
	vector<vector<int> > G;
	size_t n, nnz;
	read_edgelist(dataset, G, n, nnz, edge_n);

	int *row_offset = new int[n + 1], *col_index = new int[nnz];
	//cudaMalloc(&row_offset, sizeof(int) * n);
	//cudaMalloc(&col_index, sizeof(int) * nnz);
	int offset = 0;
	for (int i = 0; i < n; ++i) {
		memcpy(row_offset + i, &offset, sizeof(int));
		memcpy(col_index + offset, &(G[i][0]), sizeof(int) * G[i].size());
		offset += G[i].size();
	}
	row_offset[n] = nnz;
	
	TimeKeeper tk2;
	tc(n, nnz, row_offset, col_index);
	cout <<"TCTotal: " << tk2.checkTime("") << endl;   

	TimeKeeper tk1;
	pgrank(n, nnz, row_offset, col_index);
	cout <<"PageRankTotal: "<< tk1.checkTime("") << endl;  
	
	
	delete[] row_offset;
	delete[] col_index;
	//cudaFree(row_offset);
	//cudaFree(col_index);
}


vector<string> datasets;
int main(int argn, char *argv[]) {
	
	int edge_n = atoi(argv[1]);

	datasets.push_back(argv[2]);
	//datasets.push_back("livejournal");
	
	for (int i = 0; i < datasets.size(); ++i)
		test(string("../edgelists/") + datasets[i], edge_n);

}
