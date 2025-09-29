import numpy as np, time

N = 2000
A = np.random.rand(N, N)
B = np.random.rand(N, N)

start = time.time()
C = A @ B
end = time.time()

ops = 2 * N**3  # # of floating point operations in matmul
gflops = ops / (end - start) / 1e9
print(f"{gflops:.2f} GFLOPS")
