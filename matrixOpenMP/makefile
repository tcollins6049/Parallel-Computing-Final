CC = g++
NODEBUGFLAGS = -std=c++11 -fopenmp -mavx -O2 
DEBUGCFLAGS = -g -std=c++11 -fopenmp -mavx -O2

matrixMult: matrixMult.C hpc_helpers.h
	$(CC) $(DEBUGCFLAGS) matrixMult.C -o matrixMult

clean:
	rm matrixMult
	
