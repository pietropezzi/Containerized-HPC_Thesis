## Build and run Containerized MPI Application
Build the sif file using the definition file with:
```
apptainer build --fakeroot bilateral-mpi-hybrid.sif bilateral-mpi-hybrid.def
```
Once the `bilateral-mpi-hybrid.sif` has been built, execute the MPI implementation with the command:
```
mpirun -n <num_proc> apptainer exec bilateral-mpi-hybrid.sif /opt/mpi-bilateral <img> <radius> <sigma_c> <sigma_s>
```

