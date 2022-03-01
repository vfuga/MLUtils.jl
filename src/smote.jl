using MLUtils
using HDF5
A = collect(reshape(1:120, 15, 8))
h5write("./test2.h5", "mygroup2/A", A)
data = h5read("./test2.h5", "mygroup2/A", (2:3:15, 3:5))

fid = h5open("./test2.h5", "r+")

Z = collect(reshape(1:1000000, 1000, 1000))
HDF5.write(fid, "mygrp/Z1", Z)
HDF5.flush(fid)
close(fid)
fid
HDF5.create_dataset

