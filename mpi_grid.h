#include <mpi.h>

const int ndims;

int Grid_Create(int N0, int N1, int nproc, MPI_Comm* Grid_Comm, int* dims);

void Grid_GetBoundaries(int N0, int N1, int nproc, int rank, 
	MPI_Comm* Grid_Comm, int* dims,
    int* left, int* right, int* down, int* up);
    
void Grid_GetNeighbors(MPI_Comm* Grid_Comm, int* left, int* right, int* down, int* up);
