#include "mpi_grid.h"
#include <stdlib.h>
#include <mpi.h>

#define TRUE  ((int) 1)
#define FALSE ((int) 0)
#define Min(A,B) ((A)<(B)?(A):(B))

const int ndims = 2;

int IsPower(int Number)
// the function returns log_{2}(Number) if it is integer. If not it returns (-1). 
{
    unsigned int M;
    int p;
    
    if(Number <= 0)
        return(-1);
        
    M = Number; p = 0;
    while(M % 2 == 0)
    {
        ++p;
        M = M >> 1;
    }
    if((M >> 1) != 0)
        return(-1);
    else
        return(p);
    
}

int SplitFunction(int N0, int N1, int p)
// This is the splitting procedure of proc. number p. The integer p0
// is calculated such that abs(N0/p0 - N1/(p-p0)) --> min.
{
    float n0, n1;
    int p0, i;
    
    n0 = (float) N0; n1 = (float) N1;
    p0 = 0;
    
    for(i = 0; i < p; i++)
        if(n0 > n1)
        {
            n0 = n0 / 2.0;
            ++p0;
        }
        else
            n1 = n1 / 2.0;
    
    return(p0);
}

int Grid_Create(int N0, int N1, int nproc, 
        MPI_Comm* Grid_Comm, int* dims) {
    int power = IsPower(nproc);
    int p0 = SplitFunction(N0, N1, power);
    int p1 = power - p0;
    int periods[2] = {0, 0};
    
    dims[0] = (unsigned int) 1 << p0;   
    dims[1] = (unsigned int) 1 << p1;
    
    return MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, Grid_Comm);

}

void Grid_GetBoundaries(int N0, int N1, int nproc, int rank, 
        MPI_Comm* Grid_Comm, int* dims,
        int* left, int* right, int* down, int* up) {
    int power = IsPower(nproc);
    int p0 = SplitFunction(N0, N1, power);
    int p1 = power - p0;
    int n0,n1, k0,k1;
    int Coords[2];
    
    n0 = N0 >> p0;
    n1 = N1 >> p1;
    k0 = N0 - dims[0]*n0;
    k1 = N1 - dims[1]*n1;
    
    MPI_Cart_coords(*Grid_Comm, rank, ndims, Coords);
    
    *left = Coords[0] * n0 + Min(k0, Coords[0]);
    *right = (Coords[0] + 1) * n0 + Min(k0, Coords[0] + 1);
    *down = Coords[1] * n1 + Min(k1, Coords[1]);
    *up = (Coords[1] + 1) * n1 + Min(k1, Coords[1] + 1);
}

void Grid_GetNeighbors(MPI_Comm* Grid_Comm, 
        int* left, int* right, int* down, int* up) {
    MPI_Cart_shift(*Grid_Comm, 0, 1, left, right);
    MPI_Cart_shift(*Grid_Comm, 1, 1, down, up);
}


