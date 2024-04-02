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
void processTransformations(int *dCsr, int *dOffset, bool *dUpdate, int *dCumTransUp, int *dCumTransRight, int V, int E) {
	//process the transformations, returning an array telling you exactly how much each mesh should be eventually moved
	dUpdate[0] = true;
	int vertex = blockIdx.x * 1024 + threadIdx.x;
	
	if(vertex < V) {
		bool done = false, updated = false;
		do {
			//we need to avoid the warp issue so we have this convoluted way of doing the check
			done = updated;
			bool check = (!done && dUpdate[vertex]);
			int start = dOffset[vertex];
			int end = dOffset[vertex+1];
			for(int i = start; i < end; i++) {
				int neighbour = dCsr[i];
				dCumTransUp[neighbour] += check*dCumTransUp[vertex];
				dCumTransRight[neighbour] += check*dCumTransRight[vertex];
				dUpdate[neighbour] = check;
			}
			updated = check;
		} while(!done);
	} 
}

__global__
void moveMesh(int **dMesh, int *dActualTransUp, int *dActualTransRight, int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int *dFinalPng, int *dOnTop, int V, int frameSizeX, int frameSizeY){
	//moves the mesh some many places, considering the opacity of the individual elements as well

	int vertex = blockIdx.x;
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
				updated = updated | (old >= op);
			} while(!done);
		}
	}
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
	int *dCumTransUp, *dCumTransRight, *dOffset, *dCsr;
	bool *dUpdate;
	cudaMalloc(&dCumTransUp, sizeof(int) * V);
	cudaMalloc(&dCumTransRight, sizeof(int) * V);
	cudaMalloc(&dUpdate, sizeof(bool) * V);
	cudaMalloc(&dOffset, sizeof(int) * (V+1));
	cudaMalloc(&dCsr, sizeof(int) * E);
	cudaMemcpy(dCumTransUp, cumTransUp, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dCumTransRight, cumTransRight, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dOffset, hOffset, sizeof(int) * (V+1), cudaMemcpyHostToDevice);
	cudaMemcpy(dCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);

	processTransformations<<<(V+1023)/1024, min(1024, V)>>>(dCsr, dOffset, dUpdate, dCumTransUp, dCumTransRight, V, E);
	
	cudaFree(dUpdate);
	cudaFree(dCsr);
	cudaFree(dOffset);
	//now that we have the actual translations, we can move the meshes
	
	
	int *dOpacity, *dGlobalCoordinatesX, *dGlobalCoordinatesY, *dFrameSizeX, *dFrameSizeY, *dFinalPng, *dOnTop, **dMesh;
	
	cudaMalloc(&dMesh, sizeof(int*) * V);
	int* localMesh[V];

	for(int i = 0; i < V; i++) {
		int *arr;
		cudaMalloc(&arr, sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i]);
		cudaMemcpy(arr, hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i], cudaMemcpyHostToDevice);
		localMesh[i] = arr;
	}
	cudaMemcpy(dMesh, localMesh, sizeof(int*) * V, cudaMemcpyHostToDevice);

	cudaMalloc(&dOpacity, sizeof(int) * V);
	cudaMalloc(&dGlobalCoordinatesX, sizeof(int) * V);
	cudaMalloc(&dGlobalCoordinatesY, sizeof(int) * V);
	cudaMalloc(&dFrameSizeX, sizeof(int) * V);
	cudaMalloc(&dFrameSizeY, sizeof(int) * V);
	cudaMalloc(&dFinalPng, sizeof(int) * frameSizeX * frameSizeY);
	cudaMalloc(&dOnTop, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemcpy(dOpacity, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dFrameSizeX, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(dFrameSizeY, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);

	moveMesh<<<dim3(V, 100, 1), dim3(100, 1, 1)>>>(dMesh, dCumTransUp, dCumTransRight, dOpacity, dGlobalCoordinatesX, dGlobalCoordinatesY, dFrameSizeX, dFrameSizeY, dFinalPng, dOnTop, V, frameSizeX, frameSizeY);
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
