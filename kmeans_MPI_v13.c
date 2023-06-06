// -----------------------------------------------------------------------------

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <mpi.h>
#include <sys/time.h>

#define TYPE float


#define malloc2D(name, xDim, yDim) do {               \
    name = (float **)malloc(xDim * sizeof(float *));          \
    assert(name != NULL);                                   \
    name[0] = (float *)malloc(xDim * yDim * sizeof(float));   \
    assert(name[0] != NULL);                                \
    int i;for (i = 1; i < xDim; i++)                       \
    name[i] = name[i-1] + yDim;                         \
} while (0)



struct JobAmountNode{
  long sidx;
  long eidx;
  long DatasetSize;
};

double wtime(void)
{
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /* in seconds */
               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
    return now_time;
}

int      _debug = 0;

double     mpi_kmeans(TYPE**, TYPE**, int*, int, int, int, int, MPI_Comm);
double     mpi_kmeans_omp_kernel(TYPE**, TYPE**, int*, int, int, int, int, MPI_Comm);
double     mpi_kmeans_cuda_kernel(TYPE**, TYPE**, int*, int, int, int, int, MPI_Comm);
void 	mpi_read  (char* argv[],TYPE***, TYPE***, int*, int*, int*,int*, char[], struct JobAmountNode**, MPI_Comm);


void getSettings(char* argv[],int* datasetSize,int* dimensions,int* numClusters,int* numIts,char benchmarkName[]);
void initValues(TYPE ***dataset, TYPE ***centers,int datasetSize,int dimensions,int numClusters,char benchmarkName[]);
TYPE calcDist(TYPE **dataset, TYPE **centers, int a, int b,int datasetSize,int dimensions,int numClusters);
void output(TYPE **centers, int *labels,int datasetSize,int numClusters,int dimensions,char benchmarkName[]);
void sortCentersAdjustLabels(TYPE **centers, int *labels,int datasetSize,int dimensions,int numClusters);
int cmp(TYPE *aa,TYPE *bb,int dimensions);
TYPE totalDistance(TYPE **dataset, TYPE **centers, int *labels,int datasetSize,int dimensions,int numClusters);
void kmeans_pp(TYPE **dataset, TYPE **centers,int datasetSize,int dimensions, int numClusters);


void merge_array(int**labels_ult,int*labels,int dataset_size,int totalNumObjs, struct JobAmountNode*, MPI_Comm   comm);

inline int rand_int(int low,int high)
{
	double ret= (double) rand()/(double)(RAND_MAX);
	return ret*(high-low)-low;
}
inline double rand_double(double low,double high)
{
	double ret= (double) rand()/(double)(RAND_MAX);
	return ret*(high-low)-low;
}


/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
	int     i, j;

	int     totalNumObjs;

	int        rank, nproc, mpi_namelen;
	char       mpi_name[MPI_MAX_PROCESSOR_NAME];

	int datasetSize;
	int dimensions;
	int numClusters;
	int numIts;
	char benchmarkName[256];

	TYPE **dataset;
	TYPE **centers;
	int *labels;
	//////////////////////////////////////////
	int *labels_ult;
    TYPE localtotalDistance;
    TYPE totalDistance_val = 0;

	//////////////////////////////////////////

    struct JobAmountNode* JobAmountArr;
	
    MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Get_processor_name(mpi_name,&mpi_namelen);

	if (_debug) printf("Proc %d of %d running on %s\n", rank, nproc, mpi_name);

    unsetenv("CUDA_VISIBLE_DEVICES");
	MPI_Barrier(MPI_COMM_WORLD);

	/* read data points from file ------------------------------------------*/
	mpi_read(argv,&dataset,&centers, &datasetSize, &numClusters, &dimensions, &numIts ,benchmarkName,
			&JobAmountArr,MPI_COMM_WORLD);

	if (_debug) { /* print the first 4 objects' coordinates */
		int num = (datasetSize < 4) ? datasetSize : 4;
		for (i=0; i<num; i++) {
			char strline[1024], strfloat[16];
			sprintf(strline,"%d: dataset[%d]= ",rank,i);
			for (j=0; j<dimensions; j++) {
				sprintf(strfloat,"%f\t",dataset[i][j]);
				strcat(strline, strfloat);
			}
			strcat(strline, "\n");
			printf("%s",strline);
		}
		if(rank==0)
		{
			for (i=0; i<num; i++) {
				char strline_center[1024], strfloat_center[16];
				sprintf(strline_center,"%d: center[%d]= ",rank,i);
				for (j=0; j<dimensions; j++) {
					sprintf(strfloat_center,"%f\t",centers[i][j]);
					strcat(strline_center, strfloat_center);
				}
				strcat(strline_center, "\n");
				printf("%s",strline_center);
			}
		}
	}





	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
	   this array should be the same across all processes                  */
	if(rank != 0)
	{
		centers    = (TYPE**) malloc(numClusters *             sizeof(TYPE*));
		assert(centers != NULL);
		centers[0] = (TYPE*)  malloc(numClusters * dimensions * sizeof(TYPE));
		assert(centers[0] != NULL);
		for (i=1; i<numClusters; i++)
			centers[i] = centers[i-1] + dimensions;	
	}
   if(rank == 0) {
		kmeans_pp(dataset,centers,datasetSize,dimensions,numClusters);
	}
	MPI_Allreduce(&datasetSize, &totalNumObjs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	MPI_Bcast(centers[0], numClusters*dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);


	/* labels: the cluster id for each data object */
	labels = (int*) malloc(datasetSize * sizeof(int));
	assert(labels != NULL);


	/* start the core computation -------------------------------------------*/
    //system("unset NVIDIA_VISIBLE_DEVICES");
    unsetenv("CUDA_VISIBLE_DEVICES");
    //putenv("NVIDIA_VISIBLE_DEVICES=all");
	mpi_kmeans_cuda_kernel(dataset, centers, labels, dimensions, datasetSize, numClusters, 
		numIts, MPI_COMM_WORLD);
	/*-----------------------------------------------------------------------*/
   
    localtotalDistance = totalDistance(dataset,centers,labels,datasetSize,dimensions,numClusters);
    totalDistance_val = 0;

    MPI_Allreduce(&localtotalDistance,&totalDistance_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if(rank == 0)
    {
        printf("rank : %d ,totalDistance_val : %f \n",rank,totalDistance_val);
    }


	free(dataset[0]);
	free(dataset);
       
	merge_array(&labels_ult,labels,datasetSize,totalNumObjs,JobAmountArr,MPI_COMM_WORLD);

	if(rank == 0)
	{
		output(centers,labels_ult,totalNumObjs,numClusters,dimensions,benchmarkName);
	}
     
	free(labels);
  
	free(centers[0]);
	free(centers);

	MPI_Finalize();
	return(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////


// Calculate the squared Euclidean distance between dataset[a][i] and
// centers[b][i], for i = 0, ..., dimensions-1.
TYPE calcDist(TYPE **dataset, TYPE **centers, int a, int b,int datasetSize,int dimensions,int numClusters) {
	int count = 0;
	TYPE dist = 0.0;
	for (count=0; count<dimensions; count++) {
		TYPE diff = dataset[a][count] - centers[b][count];
		TYPE diffSq = diff * diff;
		dist += diffSq;
	}
	return dist;
}


// Read this problem's settings from the .setup file
void getSettings(char* argv[],int* datasetSize,int* dimensions,int* numClusters,int* numIts,char benchmarkName[]) {
	FILE *f;
	if ((f = fopen(argv[1], "r")) == NULL) {
		printf("Cannot open settings file %s\n", argv[1]);
		printf("If you are trying the large dataset, make sure you have downloaded it from the contest website.\n");
		assert(0);
	}
	printf("Reading settings from %s.\n", argv[1]);
	assert(fscanf(f, "%d", datasetSize));
	assert(fscanf(f, "%d", dimensions));
	assert(fscanf(f, "%d", numClusters));
	assert(fscanf(f, "%d", numIts));
	assert(fscanf(f, "%s", benchmarkName));

	fclose(f);
}

// Setup and initialize the dataset, centers, and clusterSums arrays.
void initValues(TYPE ***dataset, TYPE ***centers,int datasetSize,int dimensions,int numClusters,char benchmarkName[]) {

	(*dataset) = (TYPE**) malloc(   sizeof(TYPE*) * datasetSize);
	(*centers) =	(TYPE**) malloc(  sizeof(TYPE*) * numClusters);


	int i, j;

	(*dataset)[0] =(TYPE*) malloc(datasetSize * dimensions *
			sizeof(TYPE));
	assert((*dataset)[0] != NULL);
	for (i=1; i<datasetSize; i++)
		(*dataset)[i] = (*dataset)[i-1] + dimensions;

	(*centers)[0] =(TYPE*) malloc(numClusters * dimensions *
			sizeof(TYPE));
	assert((*centers)[0] != NULL);
	for (i=1; i<numClusters; i++)
		(*centers)[i] = (*centers)[i-1] + dimensions;


	// Read the values from dataset input file
	char buf[256];
	snprintf(buf, sizeof buf, "%s.input", benchmarkName);
	FILE *f;
	if ((f = fopen(buf, "r")) == NULL ) {
		printf("Can't open file %s\n", buf);
		assert(0);
	}

	for (i=0; i < datasetSize; i++) {
		for (j=0; j < dimensions; j++) {
			float x;
			if(!fscanf(f, "%f", &x)) {
				printf("ERROR reading %s.input\n", benchmarkName);
				assert(0);
			}
			(*dataset)[i][j] = x;
		}
	}
	fclose(f);

	// Read the initial value for cluster locations
	/*snprintf(buf, sizeof buf, "%s.start", benchmarkName);
	  if ((f = fopen(buf, "r" )) == NULL ) {
	  printf("Can't open file %s\n", buf);
	  assert(0);
	  }

	  for (i=0; i < numClusters; i++) {
	  for (j=0; j < dimensions; j++) {
	  int x;
	  if (!fscanf(f, "%d", &x)) {
	  printf("ERROR reading %s.start\n", benchmarkName);
	  assert(0);
	  }

	  (*centers)[i][j] = x;
	  }
	  }
	  fclose(f);*/

}


// Post-processing: output and cleanup
// 1. Sort the outputs (for easy comparison).
// 2. Store the outputs to .centerout and .labels files
// Cleanup
void output(TYPE **centers, int *labels,int datasetSize,int numClusters,int dimensions,char benchmarkName[]) {

	int k, d;
	FILE *f;

	// Sort the centers and adjust the labels accordingly.
	sortCentersAdjustLabels(centers, labels,datasetSize,numClusters,dimensions);

	// Output the (now sorted) center locations to the .centerout file.
	char buf[256];
	snprintf(buf, sizeof buf, "%s.centerout", benchmarkName);
	if ((f = fopen(buf, "w")) == NULL) {
		printf("Can't open file %s\n", buf);
		assert(0);
	}

	for (k=0; k<numClusters; k++) {
		for (d=0; d<dimensions; d++) {
			fprintf(f, "%f ", centers[k][d]);
		}
		fprintf(f, "\n");
	}
	fclose(f);

	// Output the label values (for each input) to the .labels file.
	snprintf(buf, sizeof buf, "%s.labels", benchmarkName);
	if ((f = fopen(buf, "w")) == NULL) {
		printf("Can't open file %s\n", buf);
		assert(0);
	}
	for (k=0; k<datasetSize; k++) {
		fprintf(f, "%d\n", labels[k]);
	}
	fclose(f);

}

// A post-processing function to sort the centers for easy comparison.
// Centers are sorted by their first dimension; if two centers share the
// same first-dimension value, the next dimension is used, etc.
// Each time centers are moved, the labels are updated accordingly.
void sortCentersAdjustLabels(TYPE **centers, int *labels,int datasetSize,int numClusters,int dimensions) {
	int i,j, min;


	TYPE *tmp = (TYPE*) malloc(sizeof(TYPE)*dimensions);

	for (i=0; i<numClusters-1; i++) {
		min = i;
		for (j=i+1; j<numClusters; j++) {
			if (cmp(centers[j], centers[min],dimensions) < 0)
				min = j;
		}

		if (min != i)  {
			// swap centers[i] and centers[min]
			memcpy(tmp, centers[i], sizeof(TYPE)*dimensions);
			memcpy(centers[i], centers[min], sizeof(TYPE)*dimensions);
			memcpy(centers[min], tmp, sizeof(TYPE)*dimensions);

			// swap element labels to match
			for (j=0; j<datasetSize; j++) {
				if (labels[j] == min)
					labels[j] = i;
				else if (labels[j] == i)
					labels[j] = min;
			}
		}
	}

	free(tmp);
}

// A comparison function used in sortCentersAdjustLabels
int cmp(TYPE *aa,TYPE *bb,int dimensions) {
	int i;
	for (i=0; i<dimensions; i++) {
		if (aa[i] != bb[i])
			return aa[i] - bb[i];
	}
	return 0;
}

// A post-processing function for calculating the overall total distance
// of a given solution
TYPE totalDistance(TYPE **dataset, TYPE **centers, int *labels,int datasetSize,int dimensions,int numClusters) {
	int i;
	TYPE d = 0;
	for (i=0; i<datasetSize; i++) {
		d += calcDist(dataset, centers, i, labels[i],datasetSize,dimensions,numClusters);
	}
	return d;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

void MPI_Send_segmented(void* data,int count,MPI_Datatype datatype,int destination,int tag,MPI_Comm communicator,long segment_size)
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int mpi_namelen;
  char mpi_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(mpi_name,&mpi_namelen);
  int i;
  int lim = count/segment_size;
  int rem = count%segment_size;
  int index = 0;
  for(i = 0;i<lim;i++,index++){
     MPI_Send((TYPE*)data+index*segment_size, segment_size, datatype, destination, tag*1000+i, communicator);
  }
  if(rem){
      MPI_Send((TYPE*)data+index*segment_size, rem, datatype, destination, tag*1000+i, communicator);
  }
}
void MPI_Recv_segmented(void* data,int count,MPI_Datatype datatype,int source,int tag,MPI_Comm communicator,MPI_Status* status,long segment_size)
{
  int i;
  int lim = count/segment_size;
  int rem = count%segment_size;
  int index = 0;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int mpi_namelen;
  char mpi_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(mpi_name,&mpi_namelen);
  for(i = 0;i<lim;i++,index++){
      MPI_Recv((TYPE*)data+index*segment_size,segment_size,datatype,source,tag*1000+i,communicator,status);
  }
  if(rem){
      MPI_Recv((TYPE*)data+index*segment_size,rem,datatype,source,tag*1000+i,communicator,status);
  }
}

void AvailableListNode(){
  int  rank, nproc, mpi_namelen;
  char mpi_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Get_processor_name(mpi_name,&mpi_namelen);

  if(rank==0) printf("World Size: %d\n",nproc);
  
  printf("MPI Rank: %d, Node Name: %s\n",rank,mpi_name); 
}


void kmeans_benchmarking_MPI(int datasetSize, int numClusters, int dimensions, float *etime){
  //multi-GPU kmeans with benchmarking
  //create dataset and use the time for the consequent works
  TYPE**dataset = (TYPE**) malloc(   sizeof(TYPE*) * datasetSize);
  TYPE**centers =    (TYPE**) malloc(  sizeof(TYPE*) * numClusters);
  int *labels = (int*)malloc(sizeof(int) * datasetSize);

  int i, j;

  dataset[0] =(TYPE*) malloc(datasetSize * dimensions *
            sizeof(TYPE));
  assert((dataset)[0] != NULL);
  for (i=1; i<datasetSize; i++)
        (dataset)[i] = (dataset)[i-1] + dimensions;

  (centers)[0] =(TYPE*) malloc(numClusters * dimensions *
            sizeof(TYPE));
  assert((centers)[0] != NULL);
  for (i=1; i<numClusters; i++)
        (centers)[i] = (centers)[i-1] + dimensions;

  int numIts=100;
  
  *etime = mpi_kmeans_cuda_kernel(dataset, centers, labels, dimensions, datasetSize, numClusters, 
          numIts, MPI_COMM_WORLD);

  free(dataset[0]);
  free(dataset);

  free(labels);
  free(centers[0]);
  free(centers);
}

void DevideDatasetNode(int datasetSize,int numClusters,int dimensions, struct JobAmountNode** JobAmountArr){
  int        rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  //benchmarking and find the times
  float* TimingArr = (float*) malloc(nproc * sizeof(float)); //can be modified in other function
  //init TimingArr for test
  //TimingArr[0] = 0.2;
  //TimingArr[1] = 0.8;
  //TimingArr[2] = 1.8;
  int i;
  float TimingArrSum = 0.0;
  float elapsedtime = 0.0;
  
  //elapsedtime = rank*3+1;
  kmeans_benchmarking_MPI(datasetSize,numClusters,dimensions,&elapsedtime);
  //printf("TimingArr[%d]: %f\n",i,TimingArr[i]);
 
  MPI_Allgather(&elapsedtime, 1, MPI_FLOAT, TimingArr, 1, MPI_FLOAT,
              MPI_COMM_WORLD);
  //kmeans_benchmarking_MPI(datasetSize,numClusters,dimensions,&elapsedtime);
  //printf("TimingArr[%d]: %f\n",i,TimingArr[i]);
  if(rank==0) for (i = 0; i < nproc; i++) printf("TimingArr[%d]: %f\n",i,TimingArr[i]);
  for (i = 0; i < nproc; i++) TimingArrSum+=TimingArr[i];

  double Denom;
  for (i = 0; i < nproc; i++) {
      Denom += 1.0/TimingArr[i];
  }

  (*JobAmountArr) = (struct JobAmountNode*) malloc(nproc * sizeof(struct JobAmountNode));


  int startidx = 0;
  for (i = 0; i < nproc-1; i++) {
      (*JobAmountArr)[i].DatasetSize = (1.0/TimingArr[i])/Denom * datasetSize;
      (*JobAmountArr)[i].sidx = startidx;
      (*JobAmountArr)[i].eidx = startidx + (*JobAmountArr)[i].DatasetSize -1;
      startidx+= (*JobAmountArr)[i].DatasetSize;
  }
  (*JobAmountArr)[nproc-1].DatasetSize =  datasetSize - startidx;
  (*JobAmountArr)[nproc-1].sidx = startidx;
  (*JobAmountArr)[nproc-1].eidx = datasetSize - 1;
}


/*---< mpi_read() >----------------------------------------------------------*/
void mpi_read(	char* 	  argv[],
		TYPE***	  dataset,
		TYPE***	  centers,
		int*      datasetSize,       /* no. data objects (local) */
		int*      numClusters,       /* no. data objects (local) */
		int*      dimensions,     /* no. coordinates */
		int*      numIts,
		char	  benchmarkName[],
        struct JobAmountNode** JobAmountArr,
		MPI_Comm  comm)
{
    int        i;
    int        rank, nproc,mpi_namelen;
    MPI_Status status;
    char       mpi_name[MPI_MAX_PROCESSOR_NAME];
   
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    MPI_Get_processor_name(mpi_name,&mpi_namelen);

        if (rank == 0) {
            getSettings(argv, datasetSize, dimensions, numClusters, numIts, benchmarkName);
            initValues (dataset, centers, (*datasetSize), (*dimensions), (*numClusters), benchmarkName);
        }

        /* broadcast global datasetSize and dimensions to the rest proc */
        MPI_Bcast(datasetSize,  1, MPI_INT, 0, comm);
        MPI_Bcast(numClusters,  1, MPI_INT, 0, comm);
        MPI_Bcast(dimensions ,  1, MPI_INT, 0, comm);
        MPI_Bcast(numIts     ,  1, MPI_INT, 0, comm);
        MPI_Bcast(benchmarkName ,256, MPI_CHAR, 0, comm);


        // Node benchmarking
        AvailableListNode();
        DevideDatasetNode(*datasetSize,*numClusters,*dimensions, JobAmountArr);  
        if (rank == 0) {
            for (i = 0; i < nproc; i++) {
                printf("JobAmountArr[%d].sidx: %ld\n",i,(*JobAmountArr)[i].sidx);
                printf("JobAmountArr[%d].eidx: %ld\n",i,(*JobAmountArr)[i].eidx);
                printf("JobAmountArr[%d].DatasetSize : %ld\n",i,(*JobAmountArr)[i].DatasetSize);
            }
        }    
       
        //printf("Proc %d of %d running on %s\n", rank, nproc, mpi_name);
       

        if (rank == 0) {

            /* index is the numObjs partitioned locally in proc 0 */
            (*datasetSize) = (*JobAmountArr)[0].DatasetSize;

            /* distribute dataset[] to other processes */
            for (i=1; i<nproc; i++) {
                int msg_size = (*JobAmountArr)[i].DatasetSize;
                MPI_Send_segmented((*dataset)[(*JobAmountArr)[i].sidx], (long)((long)msg_size*(long)(*dimensions)), MPI_FLOAT,i, i, comm, 1000000);
            }

            /* reduce the dataset[] to local size */
            (*dataset)[0] = realloc((*dataset)[0],
                                 (long)((long)(*datasetSize)*(long)(*dimensions)*(long)sizeof(TYPE)));

            assert((*dataset)[0] != NULL);
            (*dataset)    = realloc((*dataset), (*datasetSize)*sizeof(TYPE*));
            assert((*dataset) != NULL);
        }
        else {
            /*  local datasetSize */
            (*datasetSize) = (*JobAmountArr)[rank].DatasetSize;

            /* allocate space for data points */
            (*dataset)    = (TYPE**)malloc((*datasetSize)            *sizeof(TYPE*));
            assert((*dataset) != NULL);
            (*dataset)[0] = (TYPE*) malloc((long)((long)(*datasetSize)*(long)(*dimensions)*(long)sizeof(TYPE)));
            assert((*dataset)[0] != NULL);
            for (i=1; i<(*datasetSize); i++)
                (*dataset)[i] = (*dataset)[i-1] + (*dimensions);
            MPI_Recv_segmented((*dataset)[0], (*datasetSize)*(*dimensions), MPI_FLOAT, 0, rank, comm, &status, 1000000);
        }


}


void merge_array(int**labels_ult,int*labels,int datasetSize, int totalNumObjs,struct JobAmountNode* JobAmountArr,MPI_Comm   comm)
{
	int        i, j, rank, nproc;
	MPI_Status status;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc); 

	if (rank == 0) { /* gather labels[] from all processes ----------*/
		(*labels_ult) = (int*) malloc (totalNumObjs*sizeof(int));
		int cpy_count = 0;

		/* first, copy local labels[] */
		for (j=0; j<datasetSize; j++) {
			(*labels_ult)[cpy_count++] = labels[j];
		}

		for (i=1; i<nproc; i++) {
			datasetSize =  JobAmountArr[i].DatasetSize;
			MPI_Recv(&((*labels_ult)[JobAmountArr[i].sidx]), datasetSize, MPI_INT, i, i, comm, &status);
		}
	}
	else {
		MPI_Send(labels, datasetSize, MPI_INT, 0, rank, comm);
	}

}


void kmeans_pp(TYPE **dataset,TYPE **centers,int datasetSize,int dimensions, int numClusters)
{
    srand(time(NULL));
	int i=0,j,l;

	TYPE* distance = (TYPE*) malloc(datasetSize*sizeof(TYPE));
	TYPE randomR;
	int randomI;
	randomR = rand_double(0.0,1.0);

	randomI = rand_int(0,datasetSize-1);
	TYPE temp;

	for (i = 0; i < dimensions; i++){
		centers[0][i] = dataset[randomI][i];
	}

	int candidate = 0;
	TYPE sum = 0;
	TYPE tempdist = 0;

	for (i = 0; i < numClusters - 1; i++){   
		sum = 0;
        #pragma omp parallel for private(i,j)
		for (j = 0; j <datasetSize  ; j++){
			if (i == 0){
				distance[j] = calcDist(dataset,centers,j, i, datasetSize, dimensions, numClusters);
			}
			else{
				tempdist = calcDist(dataset,centers,j, i, datasetSize, dimensions, numClusters);
				if(distance[j]>tempdist)
					distance[j] = tempdist;
			}
		}
        #pragma omp parallel for private(j) reduction(+:sum)
		for (j = 0; j <datasetSize ; j++){
			sum += distance[j];
		}

		randomR = rand_double(0.0,sum);
		temp = distance[0];
		candidate = 0;
		while (!(randomR < temp)){
			temp += distance[++candidate];
		}
		//#pragma omp parallel for private(i,l)
		for (l = 0; l < dimensions; l++){
			centers[i + 1][l] = dataset[candidate][l];
		}
	}
}


struct JobAmount{
  long sidx;
  long eidx;
  long DatasetSize;
};

//void AvailableList(cudaDeviceProp**DeviceSpecArr, int *NumofAvailNVidiaDev);
void DevideDataset(int *NumofAvailNVidiaDev, int datasetSize,int numClusters,int dimensions, struct JobAmount** JobAmountArr);

void CUBLAS_GEMM_warmup(float *d_A, float *d_B, float *d_C,int nr_rows_A, int nr_cols_A, int nr_cols_B, int iteration,int deviceidx);


void cuda_allocation_init(int device_id,
        TYPE**     deviceObjects,
        TYPE**     deviceClusters,
        int**      deviceMembership,
        int**      device_local_clusterCounts,
        TYPE**     device_local_clusterSums,
        TYPE**     deviceCenterSquare,
        TYPE**     deviceDistances,
        TYPE**     dimObjects,
        TYPE**     dimClusters,
        int        datasetSize,
        int        numClusters,
        int        dimensions);

void cuda_kernel_call_memcpy(
        int device_id,
        int numClusterBlocks,
        int numThreadsPerClusterBlock,
        int*      device_local_clusterCounts,
        TYPE*     device_local_clusterSums,
        TYPE*     deviceObjects,
        TYPE*      deviceClusters,
        int*       deviceMembership,
        TYPE*      deviceDistances,
        TYPE**     dimClustersMerged,
        TYPE*      deviceCenterSquare,
        int        datasetSize,
        int        numClusters,
        int        dimensions);

void cuda_cpy_centers_to_host(
        int     device_id,
        TYPE*   deviceClusters,
        TYPE**     dimClusters,
        int     datasetSize,
        int     numClusters,
        int     dimensions);

void cuda_cpy_labels_to_host(
        int     device_id,
        int*   labels,
        int*  deviceMembership,
        int     datasetSize,
        int     dimensions);

void cuda_free_reset(
        int             device_id,
        TYPE*         deviceObjects,
        TYPE*         deviceClusters,
        int*           deviceMembership,
        TYPE*         deviceCenterSquare,
        TYPE*         deviceDistances,
        TYPE*          device_local_clusterSums,
        int*           device_local_clusterCounts);

/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
double mpi_kmeans_cuda_kernel(  TYPE  **dataset,      /* in: [numObjs][numCoords] */
                   TYPE  **centers,  /*cluster centers of size [numClusters][numCoords]*/
                   int  *labels,   /* out: [numObjs] */
                   int  dimensions,    /* no. features */
                   int  datasetSize,      /* no. objects */
                   int  numClusters,  /* no. clusters */
                   int  numIts,
                   MPI_Comm  comm)
{
    ////////////////////////////////////////
    //timing report
    double timing_report_transfer_start =0;
    double timing_report_transfer_sum =0;
    double timing_report_gpu_call_start =0;
    double timing_report_gpu_call_sum =0;
    //
    ////////////////////////////////////////
    int      rank, total_numObjs;
    int  i,j,k, loop=0;
    TYPE** dimClustersMergedMPI;
    malloc2D(dimClustersMergedMPI,  numClusters,dimensions);
    memset(dimClustersMergedMPI[0], 0, dimensions * numClusters * sizeof(TYPE));
    if (_debug) MPI_Comm_rank(comm, &rank);


    /* initialize labels[] */
    for (i=0; i<datasetSize; i++) labels[i] = 0;

    MPI_Allreduce(&datasetSize, &total_numObjs, 1, MPI_INT, MPI_SUM, comm);


    int       nproc, mpi_namelen;
    char       mpi_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Get_processor_name(mpi_name,&mpi_namelen);
    printf("Proc %d of %d running on %s\n", rank, nproc, mpi_name);
    if (_debug) printf("%2d: numObjs=%d total_numObjs=%d numClusters=%d numCoords=%d\n",rank,datasetSize,total_numObjs,numClusters,dimensions);

    //find GPUs specifications
    //cudaDeviceProp *DeviceSpecArr; 
    int NumofAvailNVidiaDev; 
    //AvailableList(&DeviceSpecArr, &NumofAvailNVidiaDev);

    struct JobAmount* JobAmountArr;
    DevideDataset(&NumofAvailNVidiaDev,datasetSize,numClusters,dimensions,&JobAmountArr);

    for (i = 0; i < NumofAvailNVidiaDev; i++) {
        printf("MPI rank: %d ,JobAmountArr[%d].sidx: %ld\n",rank,i,JobAmountArr[i].sidx);
        printf("MPI rank: %d ,JobAmountArr[%d].eidx: %ld\n",rank,i,JobAmountArr[i].eidx);
        printf("MPI rank: %d ,JobAmountArr[%d].DatasetSize : %ld\n",rank,i,JobAmountArr[i].DatasetSize);
    }
    printf("dimensions: %d, MPI rank: %d\n",dimensions,rank);
    printf("datasetSize: %d, MPI rank: %d\n",datasetSize,rank);
    printf("numClusters: %d, MPI rank: %d\n",numClusters,rank);
    printf("numIts: %d, MPI rank: %d\n",numIts,rank);

    float elapsedTime;
    
    int  count;
    
    TYPE  **dimClustersMerged;
    TYPE ***dimClusters_gpuarr= (TYPE***) malloc(sizeof(TYPE**)*NumofAvailNVidiaDev);
    TYPE ***dimObjects_gpuarr= (TYPE***) malloc(sizeof(TYPE**)*NumofAvailNVidiaDev);
    TYPE **deviceObjects_gpuarr= (TYPE**) malloc(sizeof(TYPE*)*NumofAvailNVidiaDev);     //dataset
    TYPE **deviceClusters_gpuarr = (TYPE**) malloc(sizeof(TYPE*)*NumofAvailNVidiaDev);   //centers
    TYPE **deviceCenterSquare_gpuarr= (TYPE**) malloc(sizeof(TYPE*)*NumofAvailNVidiaDev);   //SquareofCenters
    TYPE **deviceDistances_gpuarr= (TYPE**) malloc(sizeof(TYPE*)*NumofAvailNVidiaDev);   //DistancesCenterAndPoints

    int **deviceMembership_gpuarr= (int**) malloc(sizeof(int*)*NumofAvailNVidiaDev);  //label   
    //int datasetSize_gpu0=datasetSize;
    
    int **labels_gpuarr= (int**) malloc(NumofAvailNVidiaDev*sizeof(int*));
    for(count=0;count<NumofAvailNVidiaDev;count++){
        labels_gpuarr[count] = (int*) calloc(JobAmountArr[count].DatasetSize,sizeof(int));
    }
    

    //  Copy dataset given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    for(count=0;count<NumofAvailNVidiaDev;count++){
        malloc2D(dimObjects_gpuarr[count], dimensions, JobAmountArr[count].DatasetSize);
        for (j = 0; j < JobAmountArr[count].DatasetSize; j++) {
            for (i = 0; i < dimensions; i++) {
                dimObjects_gpuarr[count][i][j] = dataset[j+JobAmountArr[count].sidx][i];
            }
        }
    }
     for(count=0;count<NumofAvailNVidiaDev;count++){
        malloc2D(dimClusters_gpuarr[count], numClusters, dimensions);
    }
   
    malloc2D(dimClustersMerged, numClusters, dimensions);
    for (i = 0; i < dimensions; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClustersMergedMPI[j][i] = centers[j][i];
        }
    }
    
    int** device_local_clusterCounts_gpuarr = (int**) malloc(NumofAvailNVidiaDev*sizeof(int*));
    TYPE** device_local_clusterSums_gpuarr = (TYPE**) malloc(sizeof(TYPE*)*NumofAvailNVidiaDev);

    unsigned int *numThreadsPerClusterBlock = (unsigned int*) malloc(NumofAvailNVidiaDev*sizeof(unsigned int));
    unsigned int *numClusterBlocks = (unsigned int*) malloc(NumofAvailNVidiaDev*sizeof(unsigned int));
    
    for(count=0;count<NumofAvailNVidiaDev;count++){
        numThreadsPerClusterBlock[count] = 1024;
        numClusterBlocks[count] =
            (JobAmountArr[count].DatasetSize+ numThreadsPerClusterBlock[count] - 1) / numThreadsPerClusterBlock[count];
    }
    
    // int*device_local_clusterCounts_gpu0_monitor = (int*) calloc(numClusters, sizeof(int));   
    //////////////////////////////////////////////////////////////////
    //#pragma omp parallel for private(count)
    for(count=0;count<NumofAvailNVidiaDev;count++){
        cuda_allocation_init(count,&deviceObjects_gpuarr[count],&deviceClusters_gpuarr[count],&deviceMembership_gpuarr[count],&device_local_clusterCounts_gpuarr[count],
            &device_local_clusterSums_gpuarr[count],&deviceCenterSquare_gpuarr[count],&deviceDistances_gpuarr[count],dimObjects_gpuarr[count],dimClusters_gpuarr[count],
            JobAmountArr[count].DatasetSize,numClusters,dimensions);
    }
    //////////////////////////////////////////////////////////////////////
    
    //set number of threads for omp parallel region
    omp_set_num_threads(NumofAvailNVidiaDev);

    //////////////////////////////////////////////////////////////////////
    TYPE DatasetCoef = ((TYPE)datasetSize/(TYPE)total_numObjs);


    MPI_Barrier(comm);
    double curT, global_curT=MPI_Wtime();
    #pragma omp parallel for private(count)
    for(count=0;count<NumofAvailNVidiaDev;count++){
        CUBLAS_GEMM_warmup(deviceClusters_gpuarr[count],deviceObjects_gpuarr[count],deviceDistances_gpuarr[count],numClusters,dimensions,JobAmountArr[count].DatasetSize, 500, count);
    }
    
    printf("Start, rank: %d\n",rank);

    do {
            timing_report_gpu_call_start = MPI_Wtime();
            curT = timing_report_gpu_call_start;
            //#pragma omp parallel for private(count)
            for(count=0;count<NumofAvailNVidiaDev;count++){
            cuda_kernel_call_memcpy(count,numClusterBlocks[count],numThreadsPerClusterBlock[count],
                device_local_clusterCounts_gpuarr[count],device_local_clusterSums_gpuarr[count],deviceObjects_gpuarr[count],deviceClusters_gpuarr[count],
                deviceMembership_gpuarr[count],deviceDistances_gpuarr[count],dimClustersMergedMPI,deviceCenterSquare_gpuarr[count],JobAmountArr[count].DatasetSize,
                numClusters,dimensions);
               
                cuda_cpy_centers_to_host(count,deviceClusters_gpuarr[count],dimClusters_gpuarr[count],JobAmountArr[count].DatasetSize,numClusters,dimensions);
            }
            timing_report_gpu_call_sum += (MPI_Wtime()-timing_report_gpu_call_start);

            for (i = 0; i < dimensions; i++) {
                for (j = 0; j < numClusters; j++) {
                   dimClustersMerged[j][i] = 0;
                }
            }
            for(count=0;count<NumofAvailNVidiaDev;count++){
                for (i = 0; i < dimensions; i++) {
                    for (j = 0; j < numClusters; j++) {
                        dimClustersMerged[j][i] += ((TYPE)JobAmountArr[count].DatasetSize/(TYPE)datasetSize)*dimClusters_gpuarr[count][j][i];
                    }
                }
            }
            
            #pragma omp parallel for private(i,j)
            for (i = 0; i < dimensions; i++) {
                for (j = 0; j < numClusters; j++) {
                   dimClustersMerged[j][i] *= DatasetCoef;
                }
            }

            timing_report_transfer_start = MPI_Wtime();
            MPI_Allreduce(dimClustersMergedMPI[0], dimClustersMerged[0], numClusters*dimensions,
                MPI_FLOAT, MPI_SUM, comm);
            timing_report_transfer_sum += (MPI_Wtime()-timing_report_transfer_start);

            if (_debug) {
                double maxTime;
                curT = MPI_Wtime() - curT;
                MPI_Reduce(&curT, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
                printf("%d: loop=%d time=%f sec\n",rank,loop,curT);
                if (rank == 0) printf("-----maxTime------ %d: loop=%d time=%f sec\n",rank,loop,maxTime);
                if (rank == 0) printf("-----timing_report_transfer_sum------ %d: loop=%d time=%f sec\n",rank,loop,timing_report_transfer_sum);
             }

            /* #pragma omp parallel for private(i,j)
            for (i = 0; i < dimensions; i++) {
                for (j = 0; j < numClusters; j++) {
                   dimClustersMerged[j][i] = dimClustersMergedMPI[j][i];
                }
            }*/

    /*
    for(count=0;count<NumofAvailNVidiaDev;count++){
        cudaSetDevice(count);//asign a gpu to each OMP thread
        checkCuda(cudaMemcpy(labels_gpuarr[count], deviceMembership_gpuarr[count],
                  JobAmountArr[count].DatasetSize*sizeof(int), cudaMemcpyDeviceToHost));
    }


    int mergedidx=0;
    for(count=0;count<NumofAvailNVidiaDev;count++){
        for (i = 0; i < JobAmountArr[count].DatasetSize; ++i){
           labels[mergedidx++] = labels_gpuarr[count][i];
        }
    }


    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < dimensions; j++) {
            centers[i][j] = dimClustersMerged[i][j];
        }
    }
    double z = totalDistance(dataset, centers, labels);
    printf("Total distance = %f\n", z);
   */ 
 

    } while (++loop < numIts);
    
    printf("rank: %d, loop : %d all time (microsec) : %f \n",rank,loop,MPI_Wtime() - global_curT);
    // end timing here
    /////////////////////////////////////////////////////////
    // printf("End timing\n");
    for(count=0;count<NumofAvailNVidiaDev;count++){
        cuda_cpy_labels_to_host(count,labels_gpuarr[count],deviceMembership_gpuarr[count],
           JobAmountArr[count].DatasetSize,dimensions);
    }

    
    int mergedidx=0;
    for(count=0;count<NumofAvailNVidiaDev;count++){
        for (i = 0; i < JobAmountArr[count].DatasetSize; ++i){
           labels[JobAmountArr[count].sidx+i] = labels_gpuarr[count][i];
        }
    }
            
    //MPI_Allreduce(labels, labels, datasetSize,
    //     MPI_INT, MPI_SUM, comm);
    
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < dimensions; j++) {
            centers[i][j] = dimClustersMergedMPI[i][j];
        }
    }
    
    for(count=0;count<NumofAvailNVidiaDev;count++){  
        cuda_free_reset(count,deviceObjects_gpuarr[count],deviceClusters_gpuarr[count],deviceMembership_gpuarr[count],deviceCenterSquare_gpuarr[count],
            deviceDistances_gpuarr[count],device_local_clusterSums_gpuarr[count],device_local_clusterCounts_gpuarr[count]);
        free(labels_gpuarr[count]);
        free(dimObjects_gpuarr[count][0]);
        free(dimObjects_gpuarr[count]);
        free(dimClusters_gpuarr[count][0]);
        free(dimClusters_gpuarr[count]);
    }

    free(dimClustersMergedMPI[0]);
    free(dimClustersMergedMPI);
    free(dimClustersMerged[0]);
    free(dimClustersMerged);
    free(dimClusters_gpuarr);
    free(dimObjects_gpuarr);
    free(deviceObjects_gpuarr);
    free(deviceClusters_gpuarr);
    free(deviceCenterSquare_gpuarr);
    free(deviceDistances_gpuarr);
    free(deviceMembership_gpuarr);
    free(labels_gpuarr);
    free(device_local_clusterCounts_gpuarr);
    free(device_local_clusterSums_gpuarr);
    return(timing_report_gpu_call_sum);
}
