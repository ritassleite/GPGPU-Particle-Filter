/*Rita Soares de Sousa Leite
student no: 000741372
Log-Likelihood CUDA implementation*/

#include <math.h>

__device__ void warpReduce(volatile float* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

/*Basic Functions*/

/*simple substraction implementation*/
__global__ void vec_subtract(const float A, const float* B, float* C, int N)
{	
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	while(index<N){ /*Check that index is valid*/
		C[index] = A - B[index];
		index += blockDim.x * gridDim.x; /*increment by number of threads running in the grid*/
	}	
}

/*sum of vector entried with sequential reduction*/
__global__ void sum_sequential(
    float* results, int size , const float* xs) 
{
    int index = blockIdx.x * blockDim.x*2 + threadIdx.x;
    
    extern __shared__ float shared_xs[];
   
    shared_xs[threadIdx.x] = (index < size) ? xs[index] + xs[index + blockIdx.x * blockDim.x] : 0;
    __syncthreads();

   for (unsigned int stride=blockDim.x/2; stride>32; stride>>=1) {
	if (threadIdx.x < stride) {
		shared_xs[threadIdx.x] += shared_xs[threadIdx.x+stride];
	}
	__syncthreads();
    }
    if (threadIdx.x < 32) warpReduce(shared_xs, threadIdx.x);
    if (threadIdx.x == 0)
        {results[blockIdx.x] = shared_xs[0];}
}

/*maximium entry of a vector with sequential reduction*/
__global__ void get_max(const float* w, float* max, int N){
    
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    extern __shared__ float shared_xs[];
    if(index<N){
	if(w[index]>w[index + blockIdx.x * blockDim.x ])
    		{shared_xs[threadIdx.x] = w[index];}
	else{ shared_xs[threadIdx.x] = w[index + blockDim.x*blockIdx.x]; };
    }
    else {shared_xs[threadIdx.x] = -3.40282e+038;}
    __syncthreads();

   for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
	if (threadIdx.x < stride) {
        if(shared_xs[threadIdx.x] < shared_xs[threadIdx.x+stride]){
            shared_xs[threadIdx.x] = shared_xs[threadIdx.x +stride];}
	}
	__syncthreads();
    }
    if (threadIdx.x == 0)
        max[blockIdx.x] = shared_xs[0];
}

/*Loglikelihood auxiliary functions:*/

/*Get y-hats*/
__global__ void sin(int N,
    const float* xs, 
    float* ys)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index<N){
	ys[index] = sin(xs[index]);
        
	/*index += blockDim.x * blockIdx.x;*/

    }
}
/*Get Loglikelihood for each of the J sample: this outputs the answer to question 1
input xs = y-yHat*/
__global__ void log_normal(
    const float* xs,  float mean,  float var,
    float* ws,int J,const float y_n)
{  
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    float Z = -0.5*log(2*M_PI)-0.5*log(var);
    if(index<J){
	
	float diff = y_n-sin(xs[index]) - mean;
        float arg = -0.5 * diff * diff / var; 
	ws[index] = Z + arg;
     };
}

/*Question 2: Once you have a vector of log likelihoods, compute the normalized weights*/
/*Get log likelihood weight for each of the J samples*/
__global__ void unNormalised_ws(
    const float* ws, const float max_w, float* res, int J)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index<J){res[index] = exp(ws[index] - max_w);}; /*max_w: we need to create a function that given a vector, returns the maximium value}*/
    
}
/*Normalise the log likelihood weights*/
__global__ void normalise_ws(
    float* ws, float sum, int J)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index<J){
	ws[index] = ws[index]/sum;
	 
    };
}




