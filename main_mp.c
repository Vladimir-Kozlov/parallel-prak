#include "mpi_grid.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

double Solution(double x, double y) {
    return exp(1 - x*x * y*y);
}

double RightPart(double x, double y) {
    return 2 * (x*x + y*y) * (1 - 2 * x*x * y*y) * exp(1 - x*x * y*y);
}

#define Boundary(x, y) (Solution(x, y))

#define h(nodes, i) (nodes[i+1]-nodes[i])

#define H(nodes, i) ((nodes[i+1]-nodes[i-1])*0.5)

//Create grid, but using MPI to calculate dots. Yeah. 3000 dots. MPI.
void MeshGenerate(int N0, int N1, double* nodes0, double* nodes1) {
    int i;
    int nproc, rank;
    double q = 1.5, A = 2.0, B = 2.0;
    double *XNodes, *YNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    XNodes = (double*)malloc(N0 * sizeof(double));
    YNodes = (double*)malloc(N1 * sizeof(double));
    
    #pragma omp parallel for 
    for(i=0; i<N0; i++)
        if (i % nproc == rank)
            XNodes[i] = A*(pow(1.0+i/(N0-1.0),q)-1.0)/(pow(2.0,q)-1.0);
        else
            XNodes[i] = 0;
    #pragma omp parallel for 
    for(i=0; i<N1; i++)
        if (i % nproc == rank)
            YNodes[i] = B*(pow(1.0+i/(N1-1.0),q)-1.0)/(pow(2.0,q)-1.0);
        else
            YNodes[i] = 0;
    MPI_Allreduce(XNodes, nodes0, N0, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(YNodes, nodes1, N1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    free(XNodes);
    free(YNodes);
}

void SynchronizeGridStart(double** P, MPI_Comm* Grid_Comm,
        int b_left, int b_right, int b_down, int b_up,
        int n_left, int n_right, int n_down, int n_up,
        int tag, MPI_Request* request, MPI_Status* status,
        double* buf_ls, double* buf_rs, double* buf_ds, double* buf_us,
        double* buf_lr, double* buf_rr, double* buf_dr, double* buf_ur){
    int l0, l1;
    int i, j;
    
    l0 = b_right - b_left;
    l1 = b_up - b_down;
    
    #pragma omp parallel for 
    for (i = 0; i < l1; i++) {
        buf_ls[i] = P[b_left][b_down + i];
        buf_rs[i] = P[b_right - 1][b_down + i];
    }
    #pragma omp parallel for 
    for (i = 0; i < l0; i++) {
        buf_us[i] = P[b_left + i][b_up - 1];
        buf_ds[i] = P[b_left + i][b_down];
    }
    
    MPI_Irecv(buf_lr, l1, MPI_DOUBLE, n_left, tag, *Grid_Comm, request+4);
    MPI_Irecv(buf_rr, l1, MPI_DOUBLE, n_right, tag, *Grid_Comm, request+5);
    MPI_Irecv(buf_ur, l0, MPI_DOUBLE, n_up, tag, *Grid_Comm, request+6);
    MPI_Irecv(buf_dr, l0, MPI_DOUBLE, n_down, tag, *Grid_Comm, request+7);
    
    MPI_Isend(buf_ls, l1, MPI_DOUBLE, n_left, tag, *Grid_Comm, request);
    MPI_Isend(buf_rs, l1, MPI_DOUBLE, n_right, tag, *Grid_Comm, request+1);
    MPI_Isend(buf_us, l0, MPI_DOUBLE, n_up, tag, *Grid_Comm, request+2);
    MPI_Isend(buf_ds, l0, MPI_DOUBLE, n_down, tag, *Grid_Comm, request+3);
}

void SynchronizeGridFinish(double** P, int N0, int N1, 
        int b_left, int b_right, int b_down, int b_up,
        MPI_Request* request, MPI_Status* status,
        double* buf_lr, double* buf_rr, double* buf_dr, double* buf_ur){
    int l0, l1;
    int i, j;
    
    l0 = b_right - b_left;
    l1 = b_up - b_down;
    
    MPI_Waitall(8, request, status);
    
    #pragma omp parallel for 
    for (i = 0; i < l1; i++) {
        if (b_left > 0)
            P[b_left - 1][b_down + i] = buf_lr[i];
        if (b_right < N0)
            P[b_right][b_down + i] = buf_rr[i];
    }
    #pragma omp parallel for 
    for (i = 0; i < l0; i++) {
        if (b_up < N1)
            P[b_left + i][b_up] = buf_ur[i];
        if (b_down > 0)
            P[b_left + i][b_down - 1] = buf_dr[i];
    }
}

#define Max(A,B) ((A)>(B)?(A):(B))
#define Min(A,B) ((A)<(B)?(A):(B))

double DotProd(int N0, int N1, double** P, double** Q, double** H,
        int b_left, int b_right, int b_down, int b_up,
        MPI_Comm* Grid_Comm) {
    int i, j;
    int k, p0, p1, d0, d1;
    double res = 0;

    p0 = Max(1, b_left);
    p1 = Max(1, b_down);
    d0 = Min(N0 - 1, b_right) - p0;
    d1 = Min(N1 - 1, b_up) - p1;
    #pragma omp parallel for private(i, j) reduction(+:res)
    for (k = 0; k < d0 * d1; k++) {
    	i = p0 + k / d1;
    	j = p1 + k % d1;
    	res += P[i][j] * Q[i][j] * H[i][j];
    }
    // #pragma omp parallel for  collapse(2) private(i, j) reduction(+:res)
    // for (i = Max(1, b_left); i < Min(N0 - 1, b_right); i++)
    //     for (j = Max(1, b_down); j < Min(N1 - 1, b_up); j++)
    //         res += P[i][j] * Q[i][j] * H[i][j]; //* H(XNodes, i) * H(YNodes, j);
    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);
    return res;
}

void SynchronizeGridAll(double** P, int N0, int N1, 
        MPI_Comm* Grid_Comm, int* dims) {
    int nproc, rank;
    int b_left, b_right, b_up, b_down;
    int l0, l1;
    int i, j;
    double *buf = (double*)malloc(N0 * N1 * sizeof(double));
    int k, p0, p1, d0, d1;

    MPI_Comm_size(*Grid_Comm, &nproc);
    MPI_Comm_rank(*Grid_Comm, &rank);
    Grid_GetBoundaries(N0, N1, nproc, rank, Grid_Comm, dims, 
        &b_left, &b_right, &b_down, &b_up);
    l0 = b_right - b_left;
    l1 = b_up - b_down;

    p0 = 0;
    p1 = 0;
    d0 = N0 - p0;
    d1 = N1 - p1;
    #pragma omp parallel for private(i, j)
    for (k = 0; k < d0 * d1; k++) {
    	i = p0 + k / d1;
    	j = p1 + k % d1;
        if ((i < b_left) || (i >= b_right) || (j < b_down) || (j >= b_up))
            buf[i * N1 + j] = 0;
        else buf[i * N1 + j] = P[i][j];
    }
    // #pragma omp parallel for  collapse(2) private(i, j)
    // for (i = 0; i < N0; i++)
    //     for (j = 0; j < N1; j++)
    //         if ((i < b_left) || (i >= b_right) || (j < b_down) || (j >= b_up))
    //             buf[i * N1 + j] = 0;
    //         else buf[i * N1 + j] = P[i][j];

    MPI_Allreduce(MPI_IN_PLACE, buf, N0 * N1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);

    #pragma omp parallel for private(i, j)
    for (k = 0; k < d0 * d1; k++) {
    	i = p0 + k / d1;
    	j = p1 + k % d1;
        if ((i < b_left) || (i >= b_right) || (j < b_down) || (j >= b_up))
            P[i][j] = buf[i * N1 + j];
    }
    // #pragma omp parallel for  collapse(2) private(i, j)
    // for (i = 0; i < N0; i++)
    //     for (j = 0; j < N1; j++)
    //         if ((i < b_left) || (i >= b_right) || (j < b_down) || (j >= b_up))
    //             P[i][j] = buf[i * N1 + j];
}

//#define LeftPart(dAx, dAy, d2x, d2y, i, j) \
//((dAx[i][j] - dAx[i+1][j])/d2x[i]+(dAy[i][j] - dAy[i][j+1])/d2y[j])

#define LeftPart(A, dx, dy, d2x, d2y, i, j)\
(((A[i][j]-A[i-1][j])/dx[i]-(A[i+1][j]-A[i][j])/dx[i+1])/d2x[i]\
+((A[i][j]-A[i][j-1])/dy[j]-(A[i][j+1]-A[i][j])/dy[j+1])/d2y[j])
double Opt(int N0, int N1, double* XNodes, double* YNodes, double **P, 
        MPI_Comm* Grid_Comm, int* dims, int max_iter) {
    const double eps = 1e-4;
    int nproc, rank;
    int b_left, b_right, b_down, b_up, n_left, n_right, n_down, n_up;
    int iter, i, j;
    double **g, **r, **F, **n2g, **hh;
    double *dhx, *d2hx, *dhy, *d2hy;
    double alpha, tau, sp, norm;
    int l0, l1;
    int k, p0, p1, d0, d1;
    double *buf_ls, *buf_rs, *buf_us, *buf_ds, *buf_lr, *buf_rr, *buf_ur, *buf_dr;
    MPI_Request request[8];
    MPI_Status status[8];
    MPI_Comm_size(*Grid_Comm, &nproc);
    MPI_Comm_rank(*Grid_Comm, &rank);
    Grid_GetBoundaries(N0, N1, nproc, rank, Grid_Comm, dims, 
        &b_left, &b_right, &b_down, &b_up);
    Grid_GetNeighbors(Grid_Comm, &n_left, &n_right, &n_down, &n_up);
    l0 = b_right - b_left;
    l1 = b_up - b_down;
    buf_ls = (double*)malloc(l1 * sizeof(double));
    buf_lr = (double*)malloc(l1 * sizeof(double));
    #pragma omp parallel for 
    for (i=0; i<l1; i++) buf_lr[i] = 0;
    buf_rs = (double*)malloc(l1 * sizeof(double));
    buf_rr = (double*)malloc(l1 * sizeof(double));
    #pragma omp parallel for 
    for (i=0; i<l1; i++) buf_rr[i] = 0;
    buf_us = (double*)malloc(l0 * sizeof(double));
    buf_ur = (double*)malloc(l0 * sizeof(double));
    #pragma omp parallel for 
    for (i=0; i<l0; i++) buf_ur[i] = 0;
    buf_ds = (double*)malloc(l0 * sizeof(double));
    buf_dr = (double*)malloc(l0 * sizeof(double));
    #pragma omp parallel for 
    for (i=0; i<l0; i++) buf_dr[i] = 0;
    
    SynchronizeGridStart(P, Grid_Comm, 
        b_left, b_right, b_down, b_up, n_left, n_right, n_down, n_up,
        0, request, status,
        buf_ls, buf_rs, buf_ds, buf_us, buf_lr, buf_rr, buf_dr, buf_ur);
    dhx = (double*)malloc(N0 * sizeof(double));
    dhx[0] = 0;
    #pragma omp parallel for 
    for (i=1; i<N0; i++) dhx[i] = XNodes[i] - XNodes[i-1];
    dhy = (double*)malloc(N1 * sizeof(double));
    dhy[0] = 0;
    #pragma omp parallel for 
    for (i=1; i<N1; i++) dhy[i] = YNodes[i] - YNodes[i-1];
    
    d2hx = (double*)malloc(N0 * sizeof(double));
    d2hx[0] = 0;
    d2hx[N0-1] = 0;
    #pragma omp parallel for 
    for (i=1; i<N0-1; i++) d2hx[i] = (XNodes[i+1] - XNodes[i-1])/2;
    d2hy = (double*)malloc(N1 * sizeof(double));
    d2hy[0] = 0;
    d2hy[N1-1] = 0;
    #pragma omp parallel for 
    for (i=1; i<N1-1; i++) d2hy[i] = (YNodes[i+1] - YNodes[i-1])/2;
    
    g = (double**)malloc(N0 * sizeof(double*));
    //#pragma omp parallel for 
    for (i = Max(0, b_left-1); i < Min(N0, b_right+1); i++)
        g[i] = (double*)malloc(N1 * sizeof(double));
    
    hh = (double**)malloc(N0 * sizeof(double*));
    F = (double**)malloc(N0 * sizeof(double*));
    n2g = (double**)malloc(N0 * sizeof(double*));
    r = (double**)malloc(N0 * sizeof(double*));
    
    for (i = b_left; i < b_right; i++) {
        hh[i] = (double*)malloc(N1 * sizeof(double));
        F[i] = (double*)malloc(N1 * sizeof(double));
        n2g[i] = (double*)malloc(N1 * sizeof(double));
        r[i] = (double*)malloc(N1 * sizeof(double));
        #pragma omp parallel for
        for (j = b_down; j < b_up; j++) {
            hh[i][j] = d2hx[i] * d2hy[j];
            F[i][j] = RightPart(XNodes[i], YNodes[j]);
            n2g[i][j] = 0;
        }
    }

    p0 = b_left+1;
    p1 = b_down+1;
    d0 = b_right-1 - p0;
    d1 = b_up-1 - p1;
    #pragma omp parallel for private(i, j)
    for (k = 0; k < d0 * d1; k++) {
    	i = p0 + k / d1;
    	j = p1 + k % d1;
        r[i][j] = -LeftPart(P, dhx, dhy, d2hx, d2hy, i, j)+F[i][j];
        g[i][j] = r[i][j];
    }
    // #pragma omp parallel for  collapse(2) private(i, j)
    // for (i = b_left+1; i < b_right-1; i++)
    //     for (j = b_down+1; j < b_up-1; j++){
    //         r[i][j] = -LeftPart(P, dhx, dhy, d2hx, d2hy, i, j)+F[i][j];
    //         g[i][j] = r[i][j];
    //     }
    SynchronizeGridFinish(P, N0, N1, b_left, b_right, b_down, b_up, request, status,
        buf_lr, buf_rr, buf_dr, buf_ur);
    //Calculate initial r
    i = b_left;
    #pragma omp parallel for 
    for (j = b_down; j < b_up; j++){
        r[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
            0:(-LeftPart(P, dhx, dhy, d2hx, d2hy, i, j)+F[i][j]);
        g[i][j] = r[i][j];
    }
    i = b_right-1;
    if (i > b_left)
    #pragma omp parallel for 
    for (j = b_down; j < b_up; j++){
        r[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
            0:(-LeftPart(P, dhx, dhy, d2hx, d2hy, i, j)+F[i][j]);
        g[i][j] = r[i][j];
    }
    j = b_down;
    #pragma omp parallel for 
    for (i = b_left+1; i < b_right-1; i++){
        r[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
            0:(-LeftPart(P, dhx, dhy, d2hx, d2hy, i, j)+F[i][j]);
        g[i][j] = r[i][j];
    }
    j = b_up-1;
    if (j > b_down)
    #pragma omp parallel for 
    for (i = b_left+1; i < b_right-1; i++){
        r[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
            0:(-LeftPart(P, dhx, dhy, d2hx, d2hy, i, j)+F[i][j]);
        g[i][j] = r[i][j];
    }
    sp = DotProd(N0, N1, r, r, hh, b_left, b_right, b_down, b_up, Grid_Comm);
    //Optimization
    norm = eps + 1.0;
    for (iter = 0; (iter < max_iter) && (norm >= eps); iter++) {
        SynchronizeGridStart(g, Grid_Comm, 
            b_left, b_right, b_down, b_up, n_left, n_right, n_down, n_up,
            0, request, status,
            buf_ls, buf_rs, buf_ds, buf_us, buf_lr, buf_rr, buf_dr, buf_ur);

    	#pragma omp parallel for private(i, j)
    	for (k = 0; k < d0 * d1; k++) {
    		i = p0 + k / d1;
    		j = p1 + k % d1;
        	n2g[i][j] = LeftPart(g, dhx, dhy, d2hx, d2hy, i, j);
    	}

        // #pragma omp parallel for  collapse(2) private(i, j)
        // for (i = b_left+1; i < b_right-1; i++)
        //     for (j = b_down+1; j < b_up-1; j++)
        //         n2g[i][j] = LeftPart(g, dhx, dhy, d2hx, d2hy, i, j);
        SynchronizeGridFinish(g, N0, N1, b_left, b_right, b_down, b_up, request, status,
            buf_lr, buf_rr, buf_dr, buf_ur);

        i = b_left;
        #pragma omp parallel for 
        for (j = b_down; j < b_up; j++)
            n2g[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
                0:LeftPart(g, dhx, dhy, d2hx, d2hy, i, j);
        i = b_right-1;
        if (i > b_left)
        #pragma omp parallel for 
        for (j = b_down; j < b_up; j++)
            n2g[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
                0:LeftPart(g, dhx, dhy, d2hx, d2hy, i, j);
        j = b_down;
        #pragma omp parallel for 
        for (i = b_left+1; i < b_right-1; i++)
            n2g[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
                0:LeftPart(g, dhx, dhy, d2hx, d2hy, i, j);
        j = b_up-1;
        if (j > b_down)
        #pragma omp parallel for 
        for (i = b_left+1; i < b_right-1; i++)
            n2g[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?
                0:LeftPart(g, dhx, dhy, d2hx, d2hy, i, j);

        tau = sp / DotProd(N0, N1, n2g, g, hh, b_left, b_right, b_down, b_up, Grid_Comm);

    	#pragma omp parallel for private(i, j)
    	for (k = 0; k < (d0+2) * (d1+2); k++) {
    		i = p0-1 + k / (d1+2);
    		j = p1-1 + k % (d1+2);
        	P[i][j] += tau * g[i][j];
        	r[i][j] -= tau * n2g[i][j];
    	}
        // #pragma omp parallel for  collapse(2) private(i, j)
        // for (i = b_left; i < b_right; i++)
        //     for (j = b_down; j < b_up; j++) {
        //         P[i][j] += tau * g[i][j];
        //         r[i][j] -= tau * n2g[i][j];
        //     }

        norm = fabs(tau) * sqrt(DotProd(N0, N1, g, g, hh, b_left, b_right, b_down, b_up, Grid_Comm));
        if (norm < eps)
            break;

        alpha = sp;
        sp = DotProd(N0, N1, r, r, hh, b_left, b_right, b_down, b_up, Grid_Comm);
        alpha = sp / alpha;

    	#pragma omp parallel for private(i, j)
    	for (k = 0; k < (d0+2) * (d1+2); k++) {
    		i = p0-1 + k / (d1+2);
    		j = p1-1 + k % (d1+2);
        	g[i][j] = r[i][j] + alpha * g[i][j];
    	}
        // #pragma omp parallel for  collapse(2) private(i, j)
        // for (i = b_left; i < b_right; i++)
        //     for (j = b_down; j < b_up; j++)
        //         g[i][j] = r[i][j] + alpha * g[i][j];
        //if (rank == 0) printf("%d %f %f\n",iter, norm, MPI_Wtime()-tm);
    }
    //#pragma omp parallel for 
    for (i = Max(0, b_left-1); i < Min(N0, b_right+1); i++) {
        free(g[i]);
    }
    //#pragma omp parallel for 
    for (i = b_left; i < b_right; i++) {
        free(n2g[i]);
        free(F[i]);
        free(hh[i]);
        free(r[i]);
    }
    free(g);
    free(r);
    free(n2g);
    free(F);
    free(hh);
    free(dhx);
    free(dhy);
    free(d2hx);
    free(d2hy);
    
    free(buf_ls);
    free(buf_rs);
    free(buf_us);
    free(buf_ds);
    free(buf_lr);
    free(buf_rr);
    free(buf_ur);
    free(buf_dr);
    return norm;
}

int main(int argc, char **argv)
{
    int N0, N1;                     // Mesh has N0 x N1 nodes.
    
    int ProcNum, rank;              // the number of processes and rank in communicator.
    int dims[2];                    
    int Coords[2];                  
    
    MPI_Comm Grid_Comm;             // this is a handler of a grid communicator.
    
    int b_left, b_right, b_up, b_down;      // the boundaries of the process.
    int n_left, n_right, n_up, n_down;      // the neighbours of the process.
    int i, j;
    double **P;
    double *XNodes, *YNodes;
    double r;
    int max_iter;
    double t;

    N0 = atoi(argv[1]);
    N1 = atoi(argv[2]);
    max_iter = atoi(argv[3]);
    // MPI Library is being activated ...
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&ProcNum);
    
    Grid_Create(N0, N1, ProcNum, &Grid_Comm, dims);
    MPI_Comm_rank(Grid_Comm, &rank);
    Grid_GetBoundaries(N0, N1, ProcNum, rank, &Grid_Comm, dims, 
        &b_left, &b_right, &b_down, &b_up);
    
    XNodes = (double*)malloc(N0 * sizeof(double));
    YNodes = (double*)malloc(N1 * sizeof(double));
    MeshGenerate(N0, N1, XNodes, YNodes);
    
    P = (double**)malloc(N0 * sizeof(double*));
    for (i = 0; i < N0; i++)
        P[i] = (double*)malloc(N1 * sizeof(double));
    for (i = b_left; i < b_right; i++)
        for (j = b_down; j < b_up; j++)
            P[i][j] = ((i==0)||(i==N0-1)||(j==0)||(j==N1-1))?(Boundary(XNodes[i], YNodes[j])):(0);
    t = MPI_Wtime();
    r = Opt(N0, N1, XNodes, YNodes, P, &Grid_Comm, dims, max_iter);
    t = MPI_Wtime() - t;
    SynchronizeGridAll(P, N0, N1, &Grid_Comm, dims);
    if (rank == 0) {
        double s = 0;
        for (i = 0; i < N0; i++)
            for (j = 0; j < N1; j++)
                s += (P[i][j] - Solution(XNodes[i], YNodes[j]))*(P[i][j] - Solution(XNodes[i], YNodes[j])) * H(XNodes, i)*H(YNodes, j);
        printf("%f %f %f\n", t, r, sqrt(s));
        for (i = 0; i < N0; i++) {
            for (j = 0; j < N1; j++)
                printf("%f ", P[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    // The end of MPI session ...
    
    for (i = 0; i < N0; i++)
        free(P[i]);
    free(P);
    free(XNodes);
    free(YNodes);
    
    return 0;
}

