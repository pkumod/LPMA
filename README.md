# LPMA
The code for Leveled Packed Memory Array for GPU

Accepted by TKDE 2023([pdf](https://ieeexplore.ieee.org/abstract/document/10058017))

## Dataset
LPMA uses simple edgelists as input. For example:

```
vertex_number edge_number
1 2
1 3
1 4
......
```

For a graph of N edges, the edgelist file contains N+1 lines. The first line indicates the number of vertices and edges. Each of the next N lines indicates an edge consisting of a source vertex and a destination vertex(first the source vertex, then the destination)

## Usage
To compile LPMA for edge update, use the following commands:
```
cd [your path]/LPMA
make UPDATE_LPMA
```
To run LPMA, first put the data files under a directory named edgelists. For example, the directory structure should be like:
```
./LPMA
    ...
./edgelists
    soc-flickr-growth.edgelist
```
Then the command to run LPMA should be:
```
./UPDATE_LPMA edge_number batch_size soc-flickr-growth 0
```

## Citation
Please cite us with the following Bibtex format:
```
@ARTICLE{ZouLPMA21,
  author={Zou, Lei and Zhang, Fan and Lin, Yinnian and Yu, Yanpeng},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={An Efficient Data Structure for Dynamic Graph on GPUs}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TKDE.2023.3235941}
  }
```
