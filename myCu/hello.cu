#include <stdio.h>

__global__ void hello()
{
    printf("hello from gpu block=%d thread=%d\n",
    blockIdx.x,
    threadIdx.x
    );
}

int main()
{
    hello<<<2,4>>>(); // <<< num_block, num_thread >>>

    cudaDeviceSynchronize();

    return 0;
}

