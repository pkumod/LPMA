
#pragma once

#include<sys/time.h>
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<linux/unistd.h>
#include<linux/kernel.h>
#include<fstream>
#include<omp.h>
#include<algorithm>

//typedef unsigned long off_t;

class TimeKeeper{
public:
	TimeKeeper(){
		ts = new struct timeval;
		gettimeofday(ts,NULL);
	}
	~TimeKeeper(){
		delete ts;
	}
	unsigned long long checkTime(const char* printInfo) {
		struct timeval ts_end,ts_res;
		gettimeofday(&ts_end,NULL);
		timersub(&ts_end,ts,&ts_res);
		//std::cout << printInfo << " use "<< ts_res.tv_sec << " s " << ts_res.tv_usec << " us.\n";
		memcpy(ts,&ts_end,sizeof(struct timeval));
		return ts_res.tv_usec + 1000000 * ts_res.tv_sec;
	}
private:
	struct timeval *ts;
};

struct Edge{
	unsigned int src;
	unsigned int dst;
	bool operator < (const Edge & b) const{
		return src < b.src || (src == b.src && dst < b.dst);
	}
};
typedef struct MEMPACKED {
	char name1[20];
	unsigned long memTotal;
	char name2[20];
	unsigned long memFree;
	char name3[20];
	unsigned long buffers;
	char name4[20];
	unsigned long cached;
	char name5[20];
	unsigned long swapCached;
}MEM_OCCUPY;

void get_memoccupy(MEMPACKED * mem){
	FILE *fd;
	char buff[256];
	MEM_OCCUPY *m;
	m = mem;
	fd = fopen("/proc/meminfo","r");
	fgets(buff,sizeof(buff),fd);
	sscanf(buff, "%s %lu ", m->name1, &m->memTotal);
	fgets(buff,sizeof(buff),fd);
	sscanf(buff, "%s %lu ", m->name2, &m->memFree);
	fgets(buff,sizeof(buff),fd);
	sscanf(buff, "%s %lu ", m->name3, &m->buffers);
	fgets(buff,sizeof(buff),fd);
	sscanf(buff, "%s %lu ", m->name4, &m->cached);
	fgets(buff,sizeof(buff),fd);
	sscanf(buff, "%s %lu", m->name5, &m->swapCached);
	fclose(fd);
}
