/*student: Rita Leite
pendulum model state propagation CUDA implementation*/

#include <math.h>




/*
    State-space model,
    x -- hidden variable with propagation noise w,
    y -- measurements with measurement noise n,
    k -- time point

    x_{k} = f(x_{k - 1}) + w_{k}, // propagation model
    y_{k} = h(x_{k}) + n_{k}, // measurement model
*/


__global__ void pendulum_propagate(
    const float* prev_particles1,const float* prev_particles2, float dt, const float* L, const float *eps1, const float *eps2,
    float* new_particles1,float* new_particles2, int J)
{
/*
    Pendulum propagation model

    prev_particles -- array of 2d vectors, 
        first column -- position of the pendulum
        second column -- angular velocity
    dt  -- discretization step
    L   -- cholesky decomposition of the covariance matrix (Q = L * L')
    eps -- random 2d variables drawn from standard distribution eps ~ N(0, 1)
*/


    const float g = 9.8;
    
    
    int index= blockIdx.x * blockDim.x + threadIdx.x;
    if(index<J){
         float new_pos, new_vel;
         new_pos = prev_particles1[index]+dt*prev_particles2[index]+L[0]*eps1[index]+L[2]*eps2[index];
         new_vel = prev_particles2[index]-g*sin(prev_particles1[index])*dt+L[1]*eps1[index+1]+L[3]*eps2[index];
	 	
         new_particles1[index] = new_pos;
         new_particles2[index] = new_vel;
	 index+=blockIdx.x * blockDim.x ;
         }
}





