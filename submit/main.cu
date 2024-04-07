/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <unistd.h>


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


__global__
void setParents(int *dParents, int *dCsr, int *dOffset, int V){
	int vertex = blockIdx.x*1024 + threadIdx.x;
	if(vertex < V) {
		int start = dOffset[vertex];
		int end = dOffset[vertex + 1];
		for(int i=start; i < end; i++){
			dParents[dCsr[i]] = vertex;
			// printf("parent of csr i %d is %d\n", dCsr[i], vertex);
		}
	}
}

__global__
void processParent(int *dParent, volatile int *dGrandParent, int *dCumTrans, volatile int *dFinalTrans, int V, int E){
	int vertex = blockIdx.x * 1024 + threadIdx.x;
	if (vertex < V){
		int parent = dParent[vertex];
		// if (vertex == 2)
		// printf("parent of vertex %d is %d\n", vertex, parent);
		if(parent >= 0){
			// if (vertex == 2){
			// // printf("total cum is %d %d\n", dFinalTrans[parent], dFinalTrans[parent + V]);
			// // printf("current cum is %d %d\n", dCumTrans[parent], dCumTrans[parent + V]);
			// }
			dFinalTrans[vertex] += dCumTrans[parent];
			dFinalTrans[vertex + V] += dCumTrans[parent + V];
			dGrandParent[vertex] = dParent[parent];
		} else {
			dGrandParent[vertex] = -1;
		}
	}
}

__global__
void moveMesh3(int *dMesh, int *dSum, int *dActualTransUp, int *dActualTransRight, int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, volatile int *dFinalPng, volatile int *dOnTop, int V, int frameSizeX, int frameSizeY, int offset){
	//moves the mesh some many places, considering the opacity of the individual elements as well
	// printf("It laucnhed right??\n");
	int tid = blockIdx.x * 1024 + threadIdx.x;
	int vertex;
	int left = 0, right = V ;
	while(left < right){
		int mid = (left + right) / 2;
		if(dSum[mid]<= tid ){
			if(mid == V-1 || dSum[mid + 1] > tid) {
				break;
			}
			else if(dSum[mid + 1] <= tid) {
				left = mid + 1;
			}
		} else {
			right = mid;
		}
	}
	vertex = (left + right)/2;
	int off = tid - dSum[vertex];
	int r = off / dFrameSizeY[vertex];
	int c = off % dFrameSizeY[vertex];

	if(vertex < V && r < dFrameSizeX[vertex] && c < dFrameSizeY[vertex]) {
		int updatedX = dGlobalCoordinatesX[vertex] + r + dActualTransUp[vertex];
		int updatedY = dGlobalCoordinatesY[vertex] + c + dActualTransRight[vertex];
		
		if (updatedX >= 0 && updatedX < frameSizeX && updatedY >= 0 && updatedY < frameSizeY) {
			int index = updatedX * frameSizeY + updatedY;
			int op = dOpacity[vertex];
			bool done = false, updated = false;
			do {
				done = updated;
				int old = dOnTop[index];
				bool val =!(old >= 0 && old < op);
				int ret = atomicCAS((int *)&dOnTop[index], old - INT_MAX*val, -1);
				if(ret == old - INT_MAX*val) {
					dFinalPng[index] = dMesh[dSum[vertex]  + r * dFrameSizeY[vertex] + c];
					dOnTop[index] = op;
					updated = true;
				} 
				__syncthreads();
				updated = updated | (old >= op);
			} while(!done);
		}
	}
}

void checkError(int i) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error %d: %s\n", i, cudaGetErrorString(err));
} 


int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.
	int *cumTransUp = (int*) malloc (sizeof (int) * V) ;
	int *cumTransRight = (int*) malloc (sizeof (int) * V) ;
	memset (cumTransUp, 0, sizeof (int) * V) ;
	memset (cumTransRight, 0, sizeof (int) * V) ;
	for(auto &x: translations){
		if(x[1] == 0)
		cumTransUp[x[0]] -= x[2];
		else if (x[1] == 1)
		cumTransUp[x[0]] += x[2];
		else if (x[1] == 2)
		cumTransRight[x[0]] -= x[2];
		else
		cumTransRight[x[0]] += x[2];
	}
	int *dCumTrans, *dOffset, *dCsr;
	int *dFinalTrans;
	cudaMalloc(&dCumTrans, sizeof(int) * 2* V);
	cudaMalloc(&dFinalTrans, sizeof(int) * 2* V);
	cudaMalloc(&dOffset, sizeof(int) * (V+1 + E));
	dCsr = dOffset + V + 1;
	cudaMemcpy(dCumTrans, cumTransUp, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dCumTrans  + V, cumTransRight, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dFinalTrans, dCumTrans, sizeof(int) * 2* V, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dFinalTrans  + V, cumTransRight, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dOffset, hOffset, sizeof(int) * (V+1), cudaMemcpyHostToDevice);
	cudaMemcpy(dCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);
	int old[1] = {1};
	
	int *dParent;
	cudaMalloc(&dParent, sizeof(int)*V);
	setParents<<<(V + 1023)/1024, 1024>>>(dParent, dCsr, dOffset, V);
	int *dGrandParent = dOffset;
	old[0] = -1;
	cudaMemcpy(dParent, old, sizeof(int), cudaMemcpyHostToDevice);
	int log_2_x = 32 - __builtin_clz(V) - 1;
	for(int i=0; i<log_2_x + 1; i++){
		processParent<<<(V+1023)/1024, 1024>>>(dParent, dGrandParent, dCumTrans, dFinalTrans, V, E);
		cudaMemcpy(dCumTrans,  dFinalTrans,sizeof(int) * 2* V, cudaMemcpyDeviceToDevice);
		int *tmp = dParent;
		dParent = dGrandParent;
		dGrandParent = tmp;
	}
	dCumTrans = dFinalTrans;
	
	int *dFinalPng, *dOnTop; 
	int *dMesh, *dSum;
	int *dProps;
	int *sum = cumTransUp;
	sum[0] = 0;
	for(int i=1; i<V; i++) sum[i] = sum[i-1] + hFrameSizeX[i-1]*hFrameSizeY[i-1];
	int total = (sum[V-1] + hFrameSizeX[V-1]*hFrameSizeY[V-1]);
	cudaMalloc(&dMesh, sizeof(int) * (sum[V-1] + hFrameSizeX[V-1]*hFrameSizeY[V-1]));
	cudaMalloc(&dSum, sizeof(int) * V);
	
	int *hBigMesh = (int *) malloc(sizeof(int) * (sum[V-1] + hFrameSizeX[V-1]*hFrameSizeY[V-1]));
	
	for(int i = 0; i < V; i++) {
		memcpy(hBigMesh + sum[i], hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i]);
	}
	cudaMemcpy(dMesh, hBigMesh, sizeof(int) * (sum[V-1] + hFrameSizeX[V-1]*hFrameSizeY[V-1]), cudaMemcpyHostToDevice);
	cudaMemcpy(dSum, sum, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMalloc(&dProps, sizeof(int) * V * 5);
	cudaMalloc(&dFinalPng, sizeof(int) * frameSizeX * frameSizeY);
	cudaMalloc(&dOnTop, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(dFinalPng, 0, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(dOnTop, 0, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemcpy(dProps, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dProps + V, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dProps + 2*V, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dProps + 3*V, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dProps + 4*V, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);
	moveMesh3<<<(total + 1023)/1024, 1024>>>(dMesh, dSum, dCumTrans, dCumTrans + V, dProps, dProps + V,dProps + 2*V, dProps + 3*V, dProps + 4*V, dFinalPng, dOnTop, V, frameSizeX, frameSizeY, 0);
	cudaMemcpy(hFinalPng, dFinalPng, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
