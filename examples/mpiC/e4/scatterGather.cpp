#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

//---------------------------------------------------------------------------------------
//Use MPI_Scatter and MPI_Gather to compute the number of values greater than a threshold:
//Generate random numbers between 0 and 99, given a threshold value, count how many values
//are bigger than the threshold value
//---------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    int threshold = 50;

    //each process receive numElements data values from process 0
    int numElements = 3;

    //receive buffer
    int* rbuf = new int[numElements];
    int *sendbuf;
    srand (time(NULL));

    //create the data values at process 0
    if (rank == 0)
    {
        sendbuf = new int[numElements * size];
        for (int i = 0; i < size * numElements; i++)
            sendbuf[i] = rand() % 100;
    }

    //scatter the data array
    MPI_Scatter(sendbuf, numElements, MPI_INT, rbuf, numElements, MPI_INT, 0, MPI_COMM_WORLD);

    //each process print out the data values it received
    for (int i = 0; i < numElements; i++)
        printf("Process %d get data value %d\n", rank, rbuf[i]);

    //each process counts how many data values are bigger than the threshold
    int count = 0;
    for (int i = 0; i < numElements; i++)
    {
        if (rbuf[i] > threshold)
            count++;
    }
    printf("Process %d: count is %d\n", rank, count);

    //The root gather all the counts
    int* localCount = new int[size];
    MPI_Gather(&count, 1, MPI_INT,  localCount, 1,  MPI_INT, 0, MPI_COMM_WORLD);

    //The process 0 computes the total number of values bigger than the threshold
    if (rank == 0)
    {
        int totalCount = 0;
        for (int i = 0; i < size; i++)
            totalCount = totalCount + localCount[i];
        printf("Process %d: The total count is %d. \n", rank, totalCount);
    }
    MPI_Finalize();
}

