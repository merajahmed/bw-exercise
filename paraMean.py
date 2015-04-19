from mpi4py import MPI
import numpy as np
import sys

# mpirun -np 33 python paraMean.py "/Users/lxt/Downloads/isabelPressureWithHeader.raw" 4 4 2
# mpirun -np 2 python paraMean.py "/Users/lxt/Downloads/isabelPressureWithHeader.raw" 1 1 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

xPar = int(sys.argv[2])
yPar = int(sys.argv[3])
zPar = int(sys.argv[4])

xRes = 500
yRes = 500
zRes = 100

xSize = xRes / xPar
ySize = yRes / yPar
zSize = zRes / zPar

if rank == 0:
	fileName = sys.argv[1]
	data = np.fromfile(fileName, dtype=np.float32)

	data = data[3:]
	data = np.reshape(data, (xRes, yRes, zRes))

	buffer = np.empty((xSize, ySize, zSize), dtype=np.float32)
	count = 0;
	for i in range(0, xPar):
		x_min = i * xSize
		x_max = (i + 1) * xSize
		for j in range(0, yPar):
			y_min = j * ySize
			y_max = (j + 1) * ySize
			for k in range(0, zPar):
				z_min = k * zSize
				z_max = (k + 1) * zSize
				count += 1
				print (x_min, x_max, y_min, y_max, z_min, z_max, "subvolume is assigned to process " + str(count))
				buffer = data[x_min: x_max, y_min: y_max, z_min: z_max]
				buffer = np.reshape(buffer, xSize * ySize * zSize)
   				comm.Send(buffer, dest = count, tag = 0)

   	N = xPar * yPar * zPar
   	meanList = np.empty(N, dtype=np.float32)
   	for i in range (0, N):
   		a = np.empty(1, dtype=np.float32)
   		comm.Recv(a, source = (i + 1), tag = 1)
   		meanList[i] = a
   		print ("Process 0 receives local mean of " + str(i+1) + " = " + str(a[0]))
   	overallMean = np.mean(meanList)
   	print ("The overall mean = " + str(overallMean))
else:
	buffer = np.empty((xSize, ySize, zSize), dtype=np.float32)
	comm.Recv(buffer, source = 0, tag = 0)
	mean = np.mean(buffer)
	print("Process " + str(rank) + " has mean = " + str(mean))
	comm.Send(mean, dest = 0, tag = 1)


