## Build and run Containerized MPI Application
Build the sif file using the definition file with:
```
apptainer build --fakeroot bilateral-mpi-hybrid.sif bilateral-mpi-hybrid.def
```
Ed eseguire il programma containerizzato con:
```
mpirun -n <num_proc> apptainer exec bilateral-mpi-hybrid.sif /opt/mpi-bilateral <img> <radius> <sigma_c> <sigma_s>
```

