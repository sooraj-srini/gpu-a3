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
void processTransformations(int *dCsr, int *dOffset, int *dRead, int *dWrite, int *oldPos, int *newPos, int *dCumTrans, int V, int E) {
	//process the transformations, returning an array telling you exactly how much each mesh should be eventually moved
	int index = blockIdx.x*1024 + threadIdx.x;

	if(index < oldPos[0]) {
		int vertex = dRead[index];
		int start = dOffset[vertex];
		int end = dOffset[vertex + 1];
		int diff = end - start;
		int position = atomicAdd(newPos, diff);
		for(int i=0; i<diff; i++){
			int neighbor = dCsr[start + i];
			dWrite[position + i] = neighbor;
			dCumTrans[neighbor] += dCumTrans[vertex];
			dCumTrans[neighbor + V] += dCumTrans[vertex + V];
		}
	}
}

__global__
void process1(int *dCsr, int *dOffset, int *dWork, int *counter, int *dCumTrans, int V, int E) {
	int index = blockIdx.x*1024 + threadIdx.x;
	
	if(index < V && dWork[index] >= 0){
		int vertex = dWork[index];
		int start = dOffset[vertex];
		int end = dOffset[vertex + 1];
		int diff = end - start;
		int position = atomicAdd(counter, diff);
		for(int i=0; i<diff && position + i < V; i++){
			int neighbor = dCsr[start + i];
			dCumTrans[neighbor] += dCumTrans[vertex];
			dCumTrans[neighbor + V] += dCumTrans[vertex + V];
			dWork[position + i] = neighbor;
			dWork[index] = -2;
		}
	
	}
}

__global__
void processn(int *dCsr, int *dOffset, volatile int *dWork, volatile int *counter, int *dCumTrans, int V, int E) {
	int index = blockIdx.x*1024 + threadIdx.x;
	
	if(index < V){
		bool done = false, updated = false;
		do {
			done = updated;
			int vertex = dWork[index];
			if (!done && vertex >= 0) {

				int start = dOffset[vertex];
				int end = dOffset[vertex + 1];
				int diff = end - start;
				int position = atomicAdd((int *)counter, diff);
				for(int i=0; i<diff && position + i < V; i++){
					int neighbor = dCsr[start + i];
					dCumTrans[neighbor] += dCumTrans[vertex];
					dCumTrans[neighbor + V] += dCumTrans[vertex + V];
					dWork[position + i] = neighbor;
					dWork[index] = -2;
				}
				updated = true;
			}
			__syncwarp();
		} while(!done);
	
	}
}


__global__
void moveMesh(int **dMesh, int *dActualTransUp, int *dActualTransRight, int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int *dFinalPng, int *dOnTop, int V, int frameSizeX, int frameSizeY, int offset){
	//moves the mesh some many places, considering the opacity of the individual elements as well
	// printf("It laucnhed right??\n");
	int vertex = blockIdx.x + offset * ((V + 9)/10);
	int r = blockIdx.y;
	int c = threadIdx.x;

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
				int ret = atomicCAS(&dOnTop[index], old - INT_MAX*val, -1);
				if(ret == old - INT_MAX*val) {
					dFinalPng[index] = dMesh[vertex][r * dFrameSizeY[vertex] + c];
					dOnTop[index] = op;
					updated = true;
				} 
				__syncwarp();
				updated = updated | (old >= op);
			} while(!done);
		}
	}
}
__global__
void moveMesh2(int *dMesh, int *dSum, int *dActualTransUp, int *dActualTransRight, volatile int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int *dFinalPng, int *dOnTop, int V, int frameSizeX, int frameSizeY, int offset){
	//moves the mesh some many places, considering the opacity of the individual elements as well
	// printf("It laucnhed right??\n");
	int vertex = blockIdx.x + offset * ((V + 9)/10);
	int r = blockIdx.y;
	int c = threadIdx.x;

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
				__syncwarp();
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
	int *dRead, *dWrite;
	int *newPos, *oldPos;
	// cudaMalloc(&newPos, sizeof(int));
	// cudaMalloc(&oldPos, sizeof(int));
	cudaMalloc(&dCumTrans, sizeof(int) * 2* V);
	// cudaMalloc(&dCumTransRight, sizeof(int) * V);
	// cudaMalloc(&dRead, sizeof(int) * V);
	// cudaMalloc(&dWrite, sizeof(int) * V);
	// cudaMalloc(&dUpdate, sizeof(bool) * V);
	cudaMalloc(&dOffset, sizeof(int) * (V+1));
	cudaMalloc(&dCsr, sizeof(int) * E);
	cudaMemcpy(dCumTrans, cumTransUp, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dCumTrans  + V, cumTransRight, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dOffset, hOffset, sizeof(int) * (V+1), cudaMemcpyHostToDevice);
	cudaMemcpy(dCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);
	free(hOffset);
	free(hCsr);
	free(cumTransRight);
	free(cumTransUp);
	int old[1] = {1};
	// cudaMemcpy(oldPos, old, sizeof(int), cudaMemcpyHostToDevice);
	// for(int i=0; i<=V; i++) {
	// 	// printf("i %d\n", i);
	// 	processTransformations<<<(V+1023)/1024, 1024>>>(dCsr, dOffset, dRead, dWrite, oldPos, newPos, dCumTrans, V, E);
	// 	// int old[1];
	// 	cudaMemcpy(old, newPos, sizeof(int), cudaMemcpyDeviceToHost);
	// 	// printf("old value %d \n", old[0]);
	// 	if(old[0] == 0) break;
	// 	else {
	// 		int *tmp = oldPos;
	// 		oldPos = newPos;
	// 		newPos = tmp;
	// 		cudaMemset(newPos, 0, sizeof(int));
	// 		tmp = dRead;
	// 		dRead = dWrite;
	// 		dWrite = tmp;
	// 	}
	// }
	// printf("done i think\n");
	// fflush(stdout);

	
	// cudaFree(dUpdate);
	// cudaFree(dRead);
	// cudaFree(dWrite);
	// cudaFree(oldPos);
	// cudaFree(newPos);
	
	
	int *dWork, *counter;
	cudaMalloc(&dWork, sizeof(int) * V);
	cudaMalloc(&counter, sizeof(int));
	cudaMemcpy(counter, old, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(dWork, -1, sizeof(int) * V);
	old [0] = 0;
	cudaMemcpy(dWork, old, sizeof(int), cudaMemcpyHostToDevice);
	
	
	// for(int i=0; i<V; i++){
	// 	process1<<<(V+1023)/1024, 1024>>>(dCsr, dOffset, dWork, counter, dCumTrans, V, E);
	// 	cudaMemcpy(old, counter, sizeof(int), cudaMemcpyDeviceToHost);
	// 	if(old[0] == V) break;
	// }
	processn<<<(V+1023)/1024, 1024>>>(dCsr, dOffset, dWork, counter, dCumTrans, V, E);
	cudaFree(dWork);
	cudaFree(counter);
	cudaFree(dCsr);
	cudaFree(dOffset);
	// cudaMemset(counter, 0, sizeof(int));
	//now that we have the actual translations, we can move the meshes
	
	
	int *dOpacity, *dGlobalCoordinatesX, *dGlobalCoordinatesY, *dFrameSizeX, *dFrameSizeY, *dFinalPng, *dOnTop; 
	// int **dMesh;
	int *dMesh, *dSum;
	int *dProps;
	// int **localMesh = (int **) malloc(sizeof(int*)*V);
	int *sum = (int*) malloc(sizeof(int)*V);
	sum[0] = 0;
	for(int i=1; i<V; i++) sum[i] = sum[i-1] + hFrameSizeX[i-1]*hFrameSizeY[i-1];
	// for(int i=0; i<V; i++) printf("%d ", sum[i]);
	cudaMalloc(&dMesh, sizeof(int) * (sum[V-1] + hFrameSizeX[V-1]*hFrameSizeY[V-1]));
	cudaMalloc(&dSum, sizeof(int) * V);

	for(int i = 0; i < V; i++) {
		int *arr;
		// cudaMalloc(&arr, sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i]);
		cudaMemcpy(dMesh + sum[i], hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i], cudaMemcpyHostToDevice);
		free(hMesh[i]);
		// localMesh[i] = arr;
	}
	cudaMemcpy(dSum, sum, sizeof(int) * V, cudaMemcpyHostToDevice);
	free(sum);
	// cudaMemcpy(dMesh, localMesh, sizeof(int*) * V, cudaMemcpyHostToDevice);
	// printf("surely dmesh worked??\n");
	// fflush(stdout);
	cudaMalloc(&dProps, sizeof(int) * V * 5);
	// cudaMalloc(&dGlobalCoordinatesX, sizeof(int) * V);
	// cudaMalloc(&dGlobalCoordinatesY, sizeof(int) * V);
	// cudaMalloc(&dFrameSizeX, sizeof(int) * V);
	// cudaMalloc(&dFrameSizeY, sizeof(int) * V);
	cudaMalloc(&dFinalPng, sizeof(int) * frameSizeX * frameSizeY);
	cudaMalloc(&dOnTop, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(dFinalPng, 0, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(dOnTop, 0, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemcpy(dProps, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);
	free(hOpacity);
	cudaMemcpy(dProps + V, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	free(hGlobalCoordinatesX);
	cudaMemcpy(dProps + 2*V, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);
	free(hGlobalCoordinatesY);
	cudaMemcpy(dProps + 3*V, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);
	free(hFrameSizeX);
	cudaMemcpy(dProps + 4*V, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);
	free(hFrameSizeY);
	// free(hGlobalCoordinatesX);
	// free(hGlobalCoordinatesY);
	// printf("are we here yet??\n");
	// fflush(stdout);
	// sleep(60);
	// for(int i=0; i<10; i++){
	// 	moveMesh<<<dim3((V + 9)/10, 100, 1), dim3(100, 1, 1)>>>(dMesh, dCumTrans, dCumTrans + V, dOpacity, dGlobalCoordinatesX, dGlobalCoordinatesY, dFrameSizeX, dFrameSizeY, dFinalPng, dOnTop, V, frameSizeX, frameSizeY, i);
	// }
	// moveMesh<<<dim3(V, 100, 1), dim3(100, 1, 1)>>>(dMesh, dCumTrans, dCumTrans + V, dProps, dProps + V,dProps + 2*V, dProps + 3*V, dProps + 4*V, dFinalPng, dOnTop, V, frameSizeX, frameSizeY, 0);
	moveMesh2<<<dim3(V, 100, 1), dim3(100, 1, 1)>>>(dMesh, dSum, dCumTrans, dCumTrans + V, dProps, dProps + V,dProps + 2*V, dProps + 3*V, dProps + 4*V, dFinalPng, dOnTop, V, frameSizeX, frameSizeY, 0);
// 	err = cudaGetLastError();
// if (err != cudaSuccess) 
//     printf("Error: %s\n", cudaGetErrorString(err));
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
