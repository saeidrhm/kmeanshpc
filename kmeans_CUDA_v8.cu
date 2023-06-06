#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <cublas_v2.h>
#include <curand.h>

#define TYPECUDA float

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

extern "C"{
// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const TYPECUDA *A, const TYPECUDA *B, TYPECUDA *C, const int m, const int k, const int n) {
    //int lda=m,ldb=k,ldc=m;
    const TYPECUDA alf = 1;
    const TYPECUDA bet = 0;
    const TYPECUDA *alpha = &alf;
    const TYPECUDA *beta = &bet;
    //cudaEvent_t start, stop;
    //float elapsedTime;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    //double start = getMicrotime();
    //std::cout << "start timing"<< std::endl;
    //cudaEventCreate(&start);
    //cudaEventRecord(start,0);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);


    //cudaEventCreate(&stop);
    // cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);

    //cudaEventElapsedTime(&elapsedTime, start,stop);
    //std::cout << "elapsed time: "<< elapsedTime << std::endl;
    // Destroy the handle
    cublasDestroy(handle);
    cudaDeviceSynchronize(); checkLastCudaError();

}
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__forceinline__ __host__ __device__ static
TYPECUDA euclid_dist_2(
                    int datasetSize,
                    int numClusters,
                    int dimensions,
                    TYPECUDA  *objects,     // [numCoords][numObjs]
                    TYPECUDA  *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    TYPECUDA ans=0;
    for (i = 0; i < dimensions; i++) {
        ans += ((objects[dimensions*objectId  + i] - clusters[dimensions*clusterId + i])
                * objects[dimensions*objectId  + i] - clusters[dimensions*clusterId + i]);
    }

    return(ans);
}


/*----< find_nearest_cluster() >---------------------------------------------*/
 __global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          TYPECUDA *  objects,           //  [numCoords][numObjs]
                          TYPECUDA *  deviceClusters,    //  [numCoords][numClusters]
                          int *membership         //  [numObjs]
                          )
{
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if (objectId < numObjs) {
        int   index, i;
        TYPECUDA dist, min_dist;


        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numObjs,numClusters,numCoords,objects, deviceClusters, objectId, 0);

        for (i=1; i<numClusters; ++i) {
            dist = euclid_dist_2(numObjs,numClusters,numCoords,objects, deviceClusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;
        }
}

/*----< find_minimum_and_labeling() >---------------------------------------------*/
 __global__ static
void find_nearest_and_labeling(int numCoords,
                          int numObjs,
                          int numClusters,
                          TYPECUDA * objects,
                          TYPECUDA *  deviceCenterSquare,           //  
                          TYPECUDA * deviceDistances,    //
                          int *membership,         //  [numObjs]
                          int *device_local_clusterCounts,
                          TYPECUDA* device_local_clusterSums
                          )
{
    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if (objectId < numObjs) {
        int   index, i;
        TYPECUDA dist, min_dist;


        /* find the cluster id that has min distance to object */
        index    = 0;
        //long base = objectId*numClusters;
        min_dist = deviceCenterSquare[0]-2*deviceDistances[objectId];

        for (i=1; i<numClusters; ++i) {
            dist = deviceCenterSquare[i]-2*deviceDistances[numObjs*i+objectId];
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        atomicAdd(&device_local_clusterCounts[index],1);

        for (i=0; i<numCoords; i++)
            {
                atomicAdd(&device_local_clusterSums[numClusters * i + index],objects[numObjs*i+objectId]);
            }
       }
}


__global__ static
void  update_centers(int numCoords,
                     int numClusters,
                     int *device_local_clusterCounts,
                     TYPECUDA* device_local_clusterSums,
                     TYPECUDA* deviceCluster)
{
int n = blockIdx.x * blockDim.x + threadIdx.x;
int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < numClusters && m < numCoords) {
       deviceCluster[n*numCoords+m] = device_local_clusterSums[numClusters * m + n]/(TYPECUDA)device_local_clusterCounts[n];
    }
}


extern "C"{
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
    cudaDeviceSynchronize(); checkLastCudaError();
}
}

extern "C"{
void CUBLAS_GEMM_warmup(float *d_A, float *d_B, float *d_C,int nr_rows_A, int nr_cols_A, int nr_cols_B, int iteration,int deviceidx){
  cudaSetDevice(deviceidx);
  int count;
  for(count=0;count<iteration;count++)
  //printf("starting mmul \n");
      gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
}
}

extern "C"{
void kmeans_benchmark(int n, int k, int d, float *etime,int deviceidx){//in this version this function simply measure CUBLASS GEMM time
    // Error code to check return values for CUDA calls
    /*cudaError_t err = cudaSuccess;
    cudaSetDevice(deviceidx);
    err = cudaDeviceReset();
    if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
   */
    cudaSetDevice(deviceidx);

    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    // for simplicity we are going to use square arrays
    nr_rows_A = nr_rows_C = n;
    nr_cols_A = nr_rows_B = d;
    nr_cols_B = nr_cols_C = k;

    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C, *d_S, *d_ls;
    int *d_l,*d_lc;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
    cudaMalloc(&d_S, nr_cols_B * sizeof(float));
    cudaMalloc(&d_ls, nr_rows_B *nr_cols_B * sizeof(float));
    cudaMalloc(&d_l, nr_rows_A * sizeof(int));
    cudaMalloc(&d_lc, nr_cols_B * sizeof(int));

    // If you already have useful values in A and B you can copy them in GPU:
    // cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

    // Fill the arrays A and B on GPU with random numbers
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    GPU_fill_rand(d_S, 1, nr_cols_B);
    GPU_fill_rand(d_ls, nr_rows_B, nr_cols_B);
    //GPU_fill_rand(d_l, 1, nr_rows_A);
    //GPU_fill_rand(d_lc, 1, nr_cols_B);

    // Optionally we can copy the data back on CPU and print the arrays
    ////cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
    ////cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
    //std::cout << "A =" << std::endl;
    //print_matrix(h_A, nr_rows_A, nr_cols_A);
    //std::cout << "B =" << std::endl;
    //print_matrix(h_B, nr_rows_B, nr_cols_B);
    //printf("start warmup \n");
    CUBLAS_GEMM_warmup(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B, 500, deviceidx);
    //printf("end warmup \n");

    const unsigned int numThreadsPerClusterBlock = 1024;
    const unsigned int numClusterBlocks =
        (n + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);

    // Multiply A and B on GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    int count;
    //printf("start main mmul \n");
    for(count=0;count<50;count++){
        int curr_cluster_idx;
        for(curr_cluster_idx=0;curr_cluster_idx<k;curr_cluster_idx++)
           cublasSdot(h,d,&d_B[d*curr_cluster_idx], 1, &d_B[d*curr_cluster_idx], 1, &d_S[curr_cluster_idx]);
        gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

        find_nearest_and_labeling
           <<< numClusterBlocks, numThreadsPerClusterBlock >>>
           (d, n, k, d_A,
            d_S,  d_C, d_l,
            d_lc,d_ls);

        cudaDeviceSynchronize(); checkLastCudaError();
        dim3 dimBlock(32, 32);
        dim3 dimGrid;
        dimGrid.x = (k + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (d + dimBlock.y - 1) / dimBlock.y;

        update_centers
          <<<dimGrid,dimBlock>>>
           (d, k, d_lc,
            d_ls, d_B);

        cudaDeviceSynchronize(); checkLastCudaError();
    }
    //printf("end main mmul \n");

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    //std::cout << "C =" << std::endl;
    //print_matrix(h_C, nr_rows_C, nr_cols_C);

    cudaEventElapsedTime(etime, start,stop);
    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);
}
}

extern "C"{
struct JobAmount{
  long sidx;
  long eidx;
  long DatasetSize;
};
}

extern "C"{
void AvailableList(cudaDeviceProp**DeviceSpecArr, int *NumofAvailNVidiaDev){
  cudaGetDeviceCount(NumofAvailNVidiaDev);
  printf("NumofAvailNVidiaDev: %d\n",*NumofAvailNVidiaDev);
  (*DeviceSpecArr) = (cudaDeviceProp*) malloc((*NumofAvailNVidiaDev)* sizeof(cudaDeviceProp));
  int i;
  for (i = 0; i < (*NumofAvailNVidiaDev); i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&(*DeviceSpecArr)[i], i);
      printf("   Device name: %s\n", (*DeviceSpecArr)[i].name);
      printf("   Device totalGlobalMem: %ld\n", (*DeviceSpecArr)[i].totalGlobalMem);
  }
}
}

extern "C"{
void DevideDataset(int *NumofAvailNVidiaDev, int datasetSize,int numClusters,int dimensions, struct JobAmount** JobAmountArr){
  cudaDeviceProp*DeviceSpecArr;
  AvailableList(&DeviceSpecArr,NumofAvailNVidiaDev);


  //benchmarking and find the times
  float* TimingArr = (float*) malloc(*NumofAvailNVidiaDev * sizeof(float)); //can be modified in other function
  //init TimingArr for test
  //TimingArr[0] = 0.2;
  //TimingArr[1] = 0.8;
  //TimingArr[2] = 1.8;
  int i;
  float TimingArrSum = 0.0;
  for (i = 0; i < *NumofAvailNVidiaDev; i++) {
      kmeans_benchmark(datasetSize,numClusters,dimensions,&TimingArr[i],i);
      printf("TimingArr[%d]: %f\n",i,TimingArr[i]);
      TimingArrSum+=TimingArr[i];
  }

  double Denom;

  for (i = 0; i < *NumofAvailNVidiaDev; i++) {
      Denom += 1.0/TimingArr[i];
  }

  (*JobAmountArr) = (struct JobAmount*) malloc(*NumofAvailNVidiaDev * sizeof(struct JobAmount));


  int startidx = 0;
  for (i = 0; i < *NumofAvailNVidiaDev-1; i++) {
      (*JobAmountArr)[i].DatasetSize = (1.0/TimingArr[i])/Denom * datasetSize;
      (*JobAmountArr)[i].sidx = startidx;
      (*JobAmountArr)[i].eidx = startidx + (*JobAmountArr)[i].DatasetSize -1;
      startidx+= (*JobAmountArr)[i].DatasetSize;
  }
  (*JobAmountArr)[*NumofAvailNVidiaDev-1].DatasetSize =  datasetSize - startidx;
  (*JobAmountArr)[*NumofAvailNVidiaDev-1].sidx = startidx;
  (*JobAmountArr)[*NumofAvailNVidiaDev-1].eidx = datasetSize - 1;
}
}

extern "C"{
void cuda_allocation_init(int device_id,
        TYPECUDA**     deviceObjects,
        TYPECUDA**     deviceClusters,
        int**      deviceMembership,
        int**      device_local_clusterCounts,
        TYPECUDA**     device_local_clusterSums,
        TYPECUDA**     deviceCenterSquare,
        TYPECUDA**     deviceDistances,
        TYPECUDA**     dimObjects,
        TYPECUDA**     dimClusters,
        int        datasetSize,
        int        numClusters,
        int        dimensions
        )
{
       cudaError_t err = cudaSuccess;
        cudaSetDevice(device_id);//asign a gpu to each OMP thread
        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        checkCuda(cudaMalloc(deviceObjects, datasetSize*dimensions*sizeof(TYPECUDA)));
        checkCuda(cudaMalloc(deviceClusters, numClusters*dimensions*sizeof(TYPECUDA)));
        checkCuda(cudaMalloc(deviceMembership, datasetSize*sizeof(int)));
        checkCuda(cudaMemcpy((*deviceObjects), dimObjects[0],
            datasetSize*dimensions*sizeof(TYPECUDA), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc(deviceCenterSquare, numClusters*dimensions*sizeof(TYPECUDA)));
        checkCuda(cudaMalloc(deviceDistances, numClusters*datasetSize*sizeof(TYPECUDA)));
        checkCuda(cudaMalloc(device_local_clusterCounts, numClusters*sizeof(int)));
        checkCuda(cudaMalloc(device_local_clusterSums, datasetSize*dimensions*sizeof(TYPECUDA)));
        checkCuda(cudaMemcpy((*deviceClusters), dimClusters[0],
             numClusters*dimensions*sizeof(TYPECUDA), cudaMemcpyHostToDevice));
}
}

extern "C"{
void cuda_kernel_call_memcpy(
        int device_id,
        int numClusterBlocks,
        int numThreadsPerClusterBlock,
        int*      device_local_clusterCounts,
        TYPECUDA*     device_local_clusterSums,
        TYPECUDA*     deviceObjects,
        TYPECUDA*      deviceClusters,
        int*       deviceMembership,
        TYPECUDA*      deviceDistances,
        TYPECUDA**     dimClustersMerged,
        TYPECUDA*      deviceCenterSquare,
        int        datasetSize,
        int        numClusters,
        int        dimensions
        )
{
        cudaSetDevice(device_id);//asign a gpu to each OMP thread
        checkCuda(cudaMemset(device_local_clusterSums, 0.0,numClusters*dimensions*sizeof(TYPECUDA)));
        checkCuda(cudaMemset(device_local_clusterCounts, 0,numClusters*sizeof(int)));
        checkCuda(cudaMemcpy(deviceClusters, dimClustersMerged,
                    numClusters*dimensions*sizeof(TYPECUDA), cudaMemcpyHostToDevice));
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);

        //cudaEventCreate(&start);
        //cudaEventRecord(start,0);
        int curr_cluster_idx;
        for(curr_cluster_idx=0;curr_cluster_idx<numClusters;curr_cluster_idx++){
            cublasSdot(h,dimensions ,&deviceClusters[dimensions*curr_cluster_idx], 1, &deviceClusters[dimensions*curr_cluster_idx], 1, &deviceCenterSquare[curr_cluster_idx]);
        }
        gpu_blas_mmul(deviceClusters,deviceObjects,deviceDistances,numClusters,dimensions,datasetSize);
        //cudaEventCreate(&start);
        //cudaEventRecord(start,0);

        find_nearest_and_labeling
            <<< numClusterBlocks, numThreadsPerClusterBlock >>>
            (dimensions, datasetSize, numClusters, deviceObjects,
             deviceCenterSquare,  deviceDistances, deviceMembership,
              device_local_clusterCounts,device_local_clusterSums);
        cudaDeviceSynchronize(); checkLastCudaError();
        //cudaEventCreate(&stop);
        //cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);

        //cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("Elapsed time : %f ms\n" ,elapsedTime);
        dim3 dimBlock(32, 32);
        dim3 dimGrid;
        dimGrid.x = (numClusters + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (dimensions + dimBlock.y - 1) / dimBlock.y;
          update_centers
             <<<dimGrid,dimBlock>>>
             (dimensions, numClusters, device_local_clusterCounts,
              device_local_clusterSums, deviceClusters);
        cudaDeviceSynchronize(); checkLastCudaError();
}
}

extern "C"{
void cuda_cpy_centers_to_host(
        int     device_id,
        TYPECUDA*   deviceClusters,
        TYPECUDA**     dimClusters,
        int     datasetSize,
        int     numClusters,
        int     dimensions
      ){
      cudaSetDevice(device_id);//asign a gpu to each OMP thread
      checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                      numClusters*dimensions*sizeof(TYPECUDA), cudaMemcpyDeviceToHost));
}
}

extern "C" {
void cuda_cpy_labels_to_host(
        int     device_id,
        int*   labels,
        int*  deviceMembership,
        int     datasetSize,
        int     dimensions){
  
  cudaSetDevice(device_id);
  checkCuda(cudaMemcpy(labels, deviceMembership,
      datasetSize*sizeof(int), cudaMemcpyDeviceToHost));
}
}


extern "C" {
    void cuda_free_reset(
            int             device_id,
            TYPECUDA*         deviceObjects,
            TYPECUDA*         deviceClusters,
            int*           deviceMembership,
            TYPECUDA*         deviceCenterSquare,
            TYPECUDA*         deviceDistances,
            TYPECUDA*          device_local_clusterSums,
            int*           device_local_clusterCounts
            )
{
  cudaError_t err = cudaSuccess;
        cudaSetDevice(device_id);//asign a gpu to each OMP thread
        checkCuda(cudaFree(deviceObjects));
        checkCuda(cudaFree(deviceClusters));
        checkCuda(cudaFree(deviceMembership));
        checkCuda(cudaFree(deviceCenterSquare));
        checkCuda(cudaFree(deviceDistances));
        checkCuda(cudaFree(device_local_clusterCounts));
        checkCuda(cudaFree(device_local_clusterSums));

        err = cudaDeviceReset();
        if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

}
}





