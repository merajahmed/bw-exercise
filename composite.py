from mpi4py import MPI
import numpy as np
import sys
from math import ceil
from numpy import dtype, nanmax
from matplotlib.pyplot import *
from matplotlib.colors import *
# mpirun -np 33 python paraMean.py "/Users/lxt/Downloads/isabelPressureWithHeader.raw" 4 4 2
# mpirun -np 2 python paraMean.py "/Users/lxt/Downloads/isabelPressureWithHeader.raw" 1 1 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#data = np.empty()
xdiv = int(sys.argv[2])
ydiv = int(sys.argv[3])
zdiv = int(sys.argv[4])
#opacitypeak = 100
viewdim = 0
nproc = xdiv 
if ydiv>1:
    viewdim  = 1
    nproc = ydiv
elif zdiv>1:
    viewdim = 2
    nproc = zdiv

xRes = 500
yRes = 500
zRes = 100


if rank == 0:
    fileName = sys.argv[1]
    data = np.fromfile(fileName, dtype=np.float32)
    data = data[3:]
    data = np.reshape(data, (xRes, yRes, zRes))
    #opacity = np.empty(data.shape, dtype = np.float32)
    #create normalization scale for color mapping
    if viewdim != 2:
        data = np.swapaxes(data,2,viewdim)
    
    chunksize = data.shape[2]
    chunksize = ceil(chunksize/nproc)
    #buffer = np.empty((xSize, ySize, zSize), dtype=np.float32)
    print chunksize
    count = 0;
    
#eliminate nans
    data = np.nan_to_num(data)
    max_val = nanmax(data)
    min_val = np.nanmin(data)
    opacitypeak = 100
    #opacity = np.empty(data.shape, dtype = np.float32)
#create normalization scale for color mapping
    norm = Normalize(vmin = min_val, vmax=max_val)
#norm.autoscale(dimdata)
#color mapping function
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cm.hot)
    for i in range(0, nproc):
        z_min = i*chunksize
        z_max = data.shape[2] if (i+1)*chunksize > data.shape[2] else (i+1)*chunksize    
        #print (z_min, z_max, "range assigned to " + str(i+1))
        buffer = np.empty((data.shape[0], data.shape[1], z_max-z_min), dtype = np.float32)
        buffer = data[:, :, z_min:z_max]
	buffer = buffer.copy(order = 'C')
	#print buffer
        bounds = np.empty(3, dtype=np.float32)
        bounds[0] = min_val
        bounds[1] = max_val
        bounds[2] = opacitypeak
	dimensions = np.empty(3,dtype=np.int16)
	dimensions[0] = data.shape[0]
	dimensions[1] = data.shape[1]
	dimensions[2] = buffer.shape[2]
	#print buffer.shape
	#print rank,dimensions
	comm.Send(dimensions, dest = i+1, tag = 0)
        comm.Send(bounds, dest = i+1, tag = 1)
        comm.Send(buffer, dest = i+1 , tag = 2)
    Slicelist = np.empty((data.shape[0],data.shape[1],nproc), dtype=np.float32)
    for i in range (0, nproc):
        Slice = np.empty((data.shape[0],data.shape[1]), dtype=np.float32)
        comm.Recv(Slice, source = (i + 1), tag = 3)
	#print 'received slice:', i+1
        Slicelist[:,:,i] = Slice
    opacity = np.empty((data.shape[0],data.shape[1],nproc), dtype=np.float32)
    color = np.empty(Slicelist.shape,dtype=[('r', float),('g', float),('b',float)])
    #print 'colormap construction started'
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(nproc):
                dimval = Slicelist[i,j,k]
                color[i,j,k] = scalarmap.to_rgba(dimval)[:3]
		print color[i,j,k]
                if dimval >= opacitypeak:
                    opacity[i,j,k] = 1-((dimval-opacitypeak)/(max_val-opacitypeak))
                elif dimval < opacitypeak:
                    opacity[i,j,k] = 1-((dimval-opacitypeak)/(min_val-opacitypeak))
		print opacity[i,j,k]
		#print 'debug',i,j,k
    #print 'colormap constructed'
    for i in range(nproc):
        if i == 0:
            R = color[:,:,0]['r']
            G = color[:,:,0]['g']
            B = color[:,:,0]['b']
        else:
            R = np.add(np.multiply(R , (1-opacity[:,:,i])), np.multiply(color[:,:,i]['r'],opacity[:,:,i]))
            G = np.add(np.multiply(G , (1-opacity[:,:,i])), np.multiply(color[:,:,i]['g'],opacity[:,:,i]))
            B = np.add(np.multiply(B , (1-opacity[:,:,i])), np.multiply(color[:,:,i]['b'],opacity[:,:,i]))
    print 'composition completed'
#return rgb array
    C = (np.dstack((R,G,B)) * 255.999) .astype(np.uint8)
    print C
    imshow(C)
    gca().invert_yaxis()
    savefig()

        #print ("Process 0 receives local mean of " + str(i+1) + " = " + str(a[0]))
        #overallMean = np.mean(meanList)
        #print ("The overall mean = " + str(overallMean))
else:
    
    #norm.autoscale(dimdata)
    #color mapping function
    dimensions = np.empty(3, dtype= np.int16)
    comm.Recv(dimensions, source = 0, tag = 0)
    print rank, dimensions
    bounds = np.empty(3, dtype= np.float32)
    comm.Recv(bounds, source = 0, tag = 1)
    buffer = np.empty((dimensions[0],dimensions[1],dimensions[2]), dtype=np.float32)
    comm.Recv(buffer, source = 0, tag = 2)
    min_val = bounds[0]
    max_val = bounds[1]
    opacitypeak = bounds[2]
    opacity = np.empty((buffer.shape[0],buffer.shape[1],buffer.shape[2]), dtype=np.float32)
    #color = np.empty(Slicelist.shape,dtype=[('r', float),('g', float),('b',float)])
     #create transfer function
    norm = Normalize(vmin = min_val, vmax=max_val)
#norm.autoscale(dimdata)
#color mapping function
    scalarmap = cm.ScalarMappable(norm=norm, cmap=cm.hot)
    for i in range(buffer.shape[0]):
         for j in range(buffer.shape[1]):
             for k in range(buffer.shape[2]):
                 dimval = buffer[i,j,k]
                 #color[i, j, k] = scalarmap.to_rgba(dimval)[:3]
#tent function map for opacity, val = 1, max_val = 0, min_val = 0
                 if dimval >= opacitypeak:
                    opacity[i,j,k] = 1-((dimval-opacitypeak)/(max_val-opacitypeak))
                 elif dimval < opacitypeak:
                    opacity[i,j,k] = 1-((dimval-opacitypeak)/(min_val-opacitypeak))

    Slice = np.empty((buffer.shape[0],buffer.shape[1]), dtype=np.float32)
    for i in range(buffer.shape[2]):
        if i == 0:
            Slice = buffer[:,:,0]
        else:
            Slice = np.add(np.multiply(Slice , (1-opacity[:,:,i])), np.multiply(buffer[:,:,i],opacity[:,:,i]))
    #print 'sending slice:', rank 
    comm.Send(Slice, dest = 0, tag = 3)


