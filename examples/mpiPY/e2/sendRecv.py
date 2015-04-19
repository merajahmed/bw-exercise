from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# pass explicit MPI datatypes
if rank == 0:
   data = numpy.arange(10, dtype='i')
   comm.Send([data, MPI.INT], dest=1, tag=77)
   print "process 0 send: "+str(data)
elif rank == 1:
   data = numpy.empty(10, dtype='i')
   comm.Recv([data, MPI.INT], source=0, tag=77)
   print "process 1 receive: "+str(data)


# automatic MPI datatype discovery
if rank == 0:
   data = numpy.arange(10, dtype=numpy.float64)
   comm.Send(data, dest=1, tag=13)
   print "process 0 send: "+str(data)
elif rank == 1:
   data = numpy.empty(10, dtype=numpy.float64)
   comm.Recv(data, source=0, tag=13)
   print "process 1 receive: "+str(data)
