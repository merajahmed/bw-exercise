from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = (rank+1)**2
if rank==0:
   print "Before gather:"
print "Process "+str(rank)+": "+str(data)
comm.Barrier()
data = comm.gather(data, root=0)
comm.Barrier()
if rank==0:
   print "After gather:"
print "Process "+str(rank)+": "+str(data)
if rank == 0:
   for i in range(size):
       assert data[i] == (i+1)**2
else:
   assert data is None
