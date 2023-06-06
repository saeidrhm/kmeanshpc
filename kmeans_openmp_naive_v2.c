/*
 * K-means Clustering
 * MEMOCODE 2016 Design Contest
 * Peter Milder, peter.milder@stonybrook.edu
 *
 * For more information, please see README.TXT and
 * http://www.ece.stonybrook.edu/~pmilder/memocode/
 *
 * This program provides the functional reference software implementation.
 *
 * Usage: ./kmeans setupfile
 * See README.TXT for input/output specification, or run: 
 *     make runsmall
 * to compile and run the small example. 
 */

/*
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/time.h>
/**
 * Returns the current time in miliseconds.
 */
double getMicrotime(){
	struct timeval ret;
	gettimeofday(&ret, NULL);
	return ((ret.tv_sec ) * 1000000u + ret.tv_usec) / 1.e6;
}

void getSettings(char* argv[]);
void initValues(int **dataset, int **centers, long **clusterSums);
long calcDist(int **dataset, int **centers, int a, int b);
void outputAndCleanup(int **dataset, int **centers, int *labels, long **clusterSums, int *clusterCounts);
void sortCentersAdjustLabels(int **centers, int *labels);
int cmp(int *aa, int *bb);	
long totalDistance(int **dataset, int **centers, int *labels);

int datasetSize;
int dimensions;
int numClusters;
int numIts;
char benchmarkName[256];

int main(int argc, char *argv[]) {

  int it, i, k, d;
  long dist;	

  if (argc != 2) {
    printf("Usage: %s setupfile\n", argv[0]);
  }
  else {
    // Read the parameter values from the .setup file
    getSettings(argv);
		
    ///////////////////////////////////////////////////////////
    // Declare variables and call initValues to initialize them.
    //   dataset: input dataset
    //   centers: locations of the cluster centers (This is initialized
    //			based on *.start, and then the final values of this
    //          array is an output.)
    //   labels: The label (cluster number) for each of the points in the dataset
    //   clusterSums: Used to hold the sum of each of cluster's locations
    //   clusterCounts: Used to hold the number of points assigned to each cluster.

	
    int **dataset = malloc(sizeof(*dataset) * datasetSize);
    int **centers = malloc(sizeof(*centers) * numClusters);
    int *labels = malloc(sizeof(int) * datasetSize);
    long **clusterSums = malloc(sizeof(*clusterSums) * numClusters);
    int *clusterCounts = malloc(sizeof(int) * numClusters);

    // Error checking 
    assert(dataset);
    assert(centers);
    assert(labels);
    assert(clusterSums);
    assert(clusterCounts);

    // Initialize variables
    initValues(dataset, centers, clusterSums);
    
	//warm up  
	for (it=0; it<numIts; it++) {

      // Clear clusterSums and clusterCounts
      for (k=0; k<numClusters; k++) {
        for (d=0; d<dimensions; d++) {
          clusterSums[k][d] = 0;
        }
        clusterCounts[k]=0;
      }
			
      ////////////////////////////////
      // Step 1: Assign each point to a cluster
			
      // Compare all elements in dataset with all elements in centers.			
      // For each dataset element, determine which center is closest, storing
      // the closest index in labels[i] and its distance in minDist.
	  #pragma omp parallel for private(i,k,dist)
      for (i=0; i<datasetSize; i++) {
        long minDist = LONG_MAX;
        for (k=0; k<numClusters; k++) {				
          dist = calcDist(dataset, centers, i, k);
          if (dist < minDist) {
            minDist = dist;
            labels[i] = k;
          }
        }							
      }
			// Once we've found the closest cluster, add this point's location
			// to the appropriate entry in clusterSums and increment the appropriate
			// clusterCount
	        for (i=0; i<datasetSize; i++) {	
				 for (d=0; d<dimensions; d++)
						clusterSums[labels[i]][d] += dataset[i][d];
				clusterCounts[labels[i]]++;	
			}

    
    // If you would like to see the total distance after this iteration,
    // you can uncomment these lines. 
    // This code does not need to be included in the timed portion.
   	//long z = totalDistance(dataset, centers, labels);
	//printf("After iteration = %d, Total distance = %ld\n", it, z);

    }
    printf("Start timing\n");
    /////////////////////////////////////////////////////////			
    // begin timing here
	double start_time = getMicrotime();	
    // Perform numIts iterations
    for (it=0; it<numIts; it++) {

      // Clear clusterSums and clusterCounts
      for (k=0; k<numClusters; k++) {
        for (d=0; d<dimensions; d++) {
          clusterSums[k][d] = 0;
        }
        clusterCounts[k]=0;
      }
			
      ////////////////////////////////
      // Step 1: Assign each point to a cluster
			
      // Compare all elements in dataset with all elements in centers.			
      // For each dataset element, determine which center is closest, storing
      // the closest index in labels[i] and its distance in minDist.
	  
	  #pragma omp parallel for private(i,k,dist)
      for (i=0; i<datasetSize; i++) {
        long minDist = LONG_MAX;
        for (k=0; k<numClusters; k++) {				
          dist = calcDist(dataset, centers, i, k);
          if (dist < minDist) {
            minDist = dist;
            labels[i] = k;
          }
        }							
      }
			// Once we've found the closest cluster, add this point's location
			// to the appropriate entry in clusterSums and increment the appropriate
			// clusterCount
	        for (i=0; i<datasetSize; i++) {	
				 for (d=0; d<dimensions; d++)
						clusterSums[labels[i]][d] += dataset[i][d];
				clusterCounts[labels[i]]++;	
			}

			
      //////////////////////////
      // Step 2: Update cluster locations

      // New location of each cluster center is the average of all points
      // assigned to that cluster.
      for (k=0; k<numClusters; k++) {
        for (d=0; d<dimensions; d++) {
          // assert(clusterCounts[k]); 	
          centers[k][d] = clusterSums[k][d]/clusterCounts[k];
        }
      }							
    
    // If you would like to see the total distance after this iteration,
    // you can uncomment these lines. 
    // This code does not need to be included in the timed portion.
   	//long z = totalDistance(dataset, centers, labels);
	//printf("After iteration = %d, Total distance = %ld\n", it, z);

    }
    printf("Elapsed Time : %f \n",getMicrotime()-start_time);

    // end timing here
    /////////////////////////////////////////////////////////			
    printf("End timing\n");

	// Calculate the total distance
	long z = totalDistance(dataset, centers, labels);
	printf("Total distance = %ld\n", z);

    // print the output values and cleanup
    outputAndCleanup(dataset, centers, labels, clusterSums, clusterCounts);
    
  }	 
  return 0;
}

// Calculate the squared Euclidean distance between dataset[a][i] and
// centers[b][i], for i = 0, ..., dimensions-1.
long calcDist(int **dataset, int **centers, int a, int b) {
  int i, j;
  long dist = 0;
  for (i=0; i<dimensions; i++) {
    int diff = dataset[a][i] - centers[b][i];
    long diffSq = (long)diff * (long)diff;
    dist += diffSq;
  }
  return dist;
}


// Read this problem's settings from the .setup file
void getSettings(char* argv[]) {
  int z;
  FILE *f;
  if ((f = fopen(argv[1], "r")) == NULL) {
    printf("Cannot open settings file %s\n", argv[1]);
    printf("If you are trying the large dataset, make sure you have downloaded it from the contest website.\n");
    assert(0);
  }
  printf("Reading settings from %s.\n", argv[1]);
  assert(fscanf(f, "%d", &datasetSize));	
  assert(fscanf(f, "%d", &dimensions));	
  assert(fscanf(f, "%d", &numClusters));
  assert(fscanf(f, "%d", &numIts));
  assert(fscanf(f, "%s", benchmarkName));
	
  fclose(f);
}

// Setup and initialize the dataset, centers, and clusterSums arrays.
void initValues(int **dataset, int **centers, long **clusterSums) {

  int i, j, z;
  for (i=0; i<datasetSize; i++) 
    dataset[i] = malloc(sizeof(int) * dimensions);
			
  for (i=0; i<numClusters; i++) 
    centers[i] = malloc(sizeof(int) * dimensions);
				
  for (i=0; i<numClusters; i++)
    clusterSums[i] = malloc(sizeof(long) * dimensions);


  // Read the values from dataset input file
  char buf[256];
  snprintf(buf, sizeof buf, "%s.input", benchmarkName);
  FILE *f;
  if ((f = fopen(buf, "r")) == NULL ) {
    printf("Can't open file %s\n", buf);
    assert(0);
  } 
  printf("benchmarkName:%s\n",benchmarkName);  
  for (i=0; i < datasetSize; i++) {
    for (j=0; j < dimensions; j++) {
      float x;
      if(!fscanf(f, "%f", &x)) {
        printf("ERROR reading %s.input\n", benchmarkName);
        assert(0);
      }
      dataset[i][j] = x;
    }
  }
  fclose(f);
    
  // Read the initial value for cluster locations
  snprintf(buf, sizeof buf, "%s.start", benchmarkName);
  if ((f = fopen(buf, "r" )) == NULL ) {
    printf("Can't open file %s\n", buf);
    assert(0);
  } 
  
  for (i=0; i < numClusters; i++) {
    for (j=0; j < dimensions; j++) {
      float x;
      if (!fscanf(f, "%f", &x)) {
        printf("ERROR reading %s.start\n", benchmarkName);
        assert(0);
	  }

      centers[i][j] = x;
    }
  }
  fclose(f);
  
}


// Post-processing: output and cleanup
// 1. Sort the outputs (for easy comparison).
// 2. Store the outputs to .centerout and .labels files
// Cleanup 
void outputAndCleanup(int **dataset, int **centers, int *labels, long **clusterSums, int *clusterCounts) {

  int k, l, d, i;
  FILE *f;

  // Sort the centers and adjust the labels accordingly.
  sortCentersAdjustLabels(centers, labels);
	
  // Output the (now sorted) center locations to the .centerout file.
  char buf[256];
  snprintf(buf, sizeof buf, "%s.centerout", benchmarkName);
  if ((f = fopen(buf, "w")) == NULL) {
    printf("Can't open file %s\n", buf);
    assert(0);
  } 
  
  for (k=0; k<numClusters; k++) {
    for (d=0; d<dimensions; d++) {
      fprintf(f, "%d ", centers[k][d]);			
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


  // free	
  for (i=0; i<datasetSize; i++) 
    free(dataset[i]);
  free(dataset);    
  
  for (i=0; i<numClusters; i++) {
  	free(centers[i]);
  	free(clusterSums[i]);
  }
  free(centers);
  free(clusterSums);
  
  free(labels);
  free(clusterCounts);
  
}  

// A post-processing function to sort the centers for easy comparison.
// Centers are sorted by their first dimension; if two centers share the
// same first-dimension value, the next dimension is used, etc.
// Each time centers are moved, the labels are updated accordingly.
void sortCentersAdjustLabels(int **centers, int *labels) {
  int i,j, min;

  int *tmp = malloc(sizeof(int)*dimensions);

  for (i=0; i<numClusters-1; i++) {
    min = i;
    for (j=i+1; j<numClusters; j++) {
      if (cmp(centers[j], centers[min]) < 0)
        min = j;
    }
				
    if (min != i)  {
      // swap centers[i] and centers[min]
      memcpy(tmp, centers[i], sizeof(int)*dimensions);
      memcpy(centers[i], centers[min], sizeof(int)*dimensions);
      memcpy(centers[min], tmp, sizeof(int)*dimensions);
	
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
int cmp(int *aa, int *bb) {
  int i;
  for (i=0; i<dimensions; i++) {
    if (aa[i] != bb[i])
      return aa[i] - bb[i];		
  }
  return 0;
}

// A post-processing function for calculating the overall total distance
// of a given solution
long totalDistance(int **dataset, int **centers, int *labels) {
	int i;
	long d = 0;
	for (i=0; i<datasetSize; i++) {
		d += calcDist(dataset, centers, i, labels[i]);
	}
	return d;
	
}
