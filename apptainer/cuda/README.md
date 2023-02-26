## Build and run Containerized CUDA Application
Build the sif file using the definition file with:
```
apptainer build --fakeroot bilateral-cuda.sif builateral-cuda.def
```
Once the `bilateral-cuda.sif` has been built, execute the cuda implementation with the command:
```
apptainer run --nv bilateral-cuda.sif /opt/cuda-bilateral <img> <sigma_c> <sigma_s>
```
