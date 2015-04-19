from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
   data = {'key1' : [7, 2.72, 2+3j],
           'key2' : ( 'abc', 'xyz')}
else:
   data = None
print "Before broadcast:"
print "Process "+str(rank)+": "+str(data)
comm.Barrier()
data = comm.bcast(data, root=0)
comm.Barrier()
print "After broadcast:"
print "Process "+str(rank)+": "+str(data)
