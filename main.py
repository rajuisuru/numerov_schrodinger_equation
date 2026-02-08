import math

L = 1.0
a = 0.0
b = L

N = 200
h = (b - a)/N

x = [a + i*h for i in range(N+1)]

def V(x):
    return 0.0

def k(x, E):
    return 2*(E - V(x))

def numerov_step(psi_nm1, psi_n, k_nm1, k_n, k_np1, h):
    return (
        2*psi_n*(1 - 5*h*h*k_n/12)
        - psi_nm1*(1 + h*h*k_nm1/12)
    ) / (1 + h*h*k_np1/12)

def shoot_left(E):

    psi = [0.0]*(N+1)

    psi[0] = 0.0        # boundary
    psi[1] = 1e-6       # small seed

    for i in range(1, N):

        k_nm1 = k(x[i-1], E)
        k_n   = k(x[i], E)
        k_np1 = k(x[i+1], E)

        psi[i+1] = numerov_step(
            psi[i-1], psi[i],
            k_nm1, k_n, k_np1, h
        )

    return psi
E = 5.0

psi = shoot_left(E)

print("psi[0] =", psi[0])
print("psi[mid] =", psi[N//2])
print("psi[end] =", psi[-1])

