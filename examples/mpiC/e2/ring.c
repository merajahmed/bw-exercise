#include <stdio.h>
#include "mpi.h"

int main( argc, argv )
int argc;
char **argv;
{
    int rank, value, size;
    MPI_Status status;

    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    value=7;
    if (rank == 0) {
	MPI_Send( &value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD );
        printf("Process %d send message to %d\n",rank, rank+1);
    }
    else {
	MPI_Recv( &value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, 
		      &status );
        printf("Process %d receive message from %d\n",rank, rank-1);
	if (rank < size - 1)
	{
	    MPI_Send( &value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD );
	    printf("Process %d send message to %d\n", rank, rank+1);
	}
    }
    

    MPI_Finalize( );
    return 0;
}
