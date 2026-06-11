// 每只鸟追逐自己的暗恋对象
#include <stdio.h>
#include <math.h>

__global__ void update(
    float *x, float *y, float *vx, float *vy,
    float w, float h, int n, int *follow, // 暗恋对象
    float vel, float dt, float alpha)
{
    int self_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(self_idx >= n)
        return;

    float self_x = x[self_idx];
    float self_y = y[self_idx];
    float self_vx = vx[self_idx];
    float self_vy = vy[self_idx];

    int crush_idx = follow[self_idx];
    float crush_x = x[crush_idx];
    float crush_y = y[crush_idx];

    float rel_x = crush_x - self_x;
    float rel_y = crush_y - self_y;

    float rel_r = sqrtf(rel_x * rel_x + rel_y * rel_x);

    if(rel_r < 1e-6f) // 已经追到手了
        return;

    float target_vx = vel * rel_x / rel_r;
    float target_vy = vel * rel_y / rel_r;
    
    float new_vx = alpha * target_vx + (1-alpha) * self_vx;
    float new_vy = alpha * target_vy + (1-alpha) * self_vy;

    // 先更新，然后判断越界
    vx[self_idx] = new_vx;
    vy[self_idx] = new_vy;

    float new_x = self_x + new_vx * dt;
    float new_y = self_y + new_vy * dt;

    new_x = fmodf(new_x + w, w);
    new_y = fmodf(new_y + h, h);
    
    x[self_idx] = new_x;
    y[self_idx] = new_y;
}

// python  传入 x,y,vx,vy,n,w,h,vel,alpha,dt
int main()
{

    int num_block = 50;
    int num_thread = n / num_block;

    bird<<<num_block,num_thread>>>();

    cudaDeviceSynchronize();

    return 0;
}

