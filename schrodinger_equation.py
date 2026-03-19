# numerov_project_final_fixed.py
# Final project-grade Numerov solver (fixed unpacking bug)
# Systems: infinite well, harmonic oscillator, finite well
# Features: automatic eigenvalue detection, normalization, plotting,
#           convergence + error analysis, saved figures.
# Units: natural units (ħ = 1, m = 1)

import math
import matplotlib.pyplot as plt
from time import perf_counter

# ---------------------------
# Numerov engine (general)
# ---------------------------
def numerov(E, x, V):
    """Numerov propagation on grid x for potential V(x).
    Returns:
        psi: list of psi at grid points
        psi_end: psi at last grid point
    """
    N = len(x) - 1
    h = x[1] - x[0]

    psi = [0.0] * (N + 1)
    psi[0] = 0.0
    psi[1] = 1e-8  # small seed

    def k(xi):
        return 2.0 * (E - V(xi))

    for i in range(1, N):
        k_im1 = k(x[i-1])
        k_i   = k(x[i])
        k_ip1 = k(x[i+1])

        denom = 1.0 + (h*h*k_ip1)/12.0
        psi[i+1] = (
            2.0 * psi[i] * (1.0 - (5.0*h*h*k_i)/12.0)
            - psi[i-1] * (1.0 + (h*h*k_im1)/12.0)
        ) / denom

    return psi, psi[-1]

# ---------------------------
# Root finding: bisection on psi(end)
# ---------------------------
def find_energy_bisect(x, V, low, high, Niter=40):
    f_low = numerov(low, x, V)[1]
    if abs(f_low) < 1e-14:
        return low

    for _ in range(Niter):
        mid = 0.5*(low + high)
        f_mid = numerov(mid, x, V)[1]
        if f_low * f_mid <= 0:
            high = mid
        else:
            low = mid
            f_low = f_mid
    return 0.5*(low + high)

# ---------------------------
# Scan and bracket eigenvalues then refine
# ---------------------------
def find_eigenvalues(x, V, E_min, E_max, steps=400):
    energies = []
    E_vals = [E_min + i*(E_max - E_min)/steps for i in range(steps+1)]

    prev_E = E_vals[0]
    prev_val = numerov(prev_E, x, V)[1]

    for E in E_vals[1:]:
        curr_val = numerov(E, x, V)[1]
        if prev_val * curr_val < 0:
            eigen_E = find_energy_bisect(x, V, prev_E, E)
            energies.append(eigen_E)
        prev_E, prev_val = E, curr_val

    return energies

# ---------------------------
# Utilities
# ---------------------------
def create_grid(a, b, N):
    h = (b - a) / N
    return [a + i*h for i in range(N+1)], h

def normalize(psi, h):
    norm = math.sqrt(sum(p*p for p in psi) * h)
    if norm == 0:
        return psi
    return [p / norm for p in psi]

def save_plot(fig_name):
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_name}")

# ---------------------------
# Potentials
# ---------------------------
def V_infinite_well(x, L=1.0):
    # inside the grid we use V=0; grid should be restricted to [0,L]
    return 0.0

def V_harmonic(x):
    return 0.5 * x * x

def V_finite_well(x, L=1.0, V0=50.0):
    # finite well defined as -V0 inside [0,L], 0 outside
    if 0.0 <= x <= L:
        return -V0
    else:
        return 0.0

# ---------------------------
# Analytic energies (for comparison)
# ---------------------------
def analytic_E_infinite(n, L=1.0):
    return (n*n * math.pi*math.pi) / (2.0 * (L*L))

def analytic_E_harmonic(n):
    return n + 0.5

# ---------------------------
# Convergence / error helper
# ---------------------------
def convergence_table_for_known(system_name, run_params, analytic_func, target_ns, N_list):
    print(f"\n=== Convergence & Error Analysis: {system_name} ===")
    # Header
    header = f"{'N':>6} {'h':>10}"
    for n in target_ns:
        header += f" {'E_num(n='+str(n)+')':>14}"
    for n in target_ns:
        header += f" {'err(n='+str(n)+')':>14}"
    print(header)

    results = {n: [] for n in target_ns}
    hs = []

    for N in N_list:
        x, h = run_params['grid_func'](N)
        hs.append(h)
        energies = find_eigenvalues(x, run_params['V_func'], run_params['E_min'], run_params['E_max'], steps=run_params.get('steps', 600))
        for n in target_ns:
            idx = n-1
            val = energies[idx] if idx < len(energies) else None
            results[n].append(val)

    for i, N in enumerate(N_list):
        row = f"{N:6d} {hs[i]:10.6e}"
        for n in target_ns:
            E_num = results[n][i]
            if E_num is None:
                row += f" {'---':>14}"
            else:
                row += f" {E_num:14.6f}"
        for n in target_ns:
            E_num = results[n][i]
            if E_num is None:
                row += f" {'---':>14}"
            else:
                if analytic_func == analytic_E_infinite:
                    E_an = analytic_func(n, run_params.get('L',1.0))
                else:
                    E_an = analytic_func(n)
                err = abs(E_num - E_an)
                row += f" {err:14.6e}"
        print(row)

    # empirical order p using last two refinements
    print("\nEstimated empirical orders (using last two grid refinements):")
    for n in target_ns:
        vals = results[n]
        pairs = [(hs[i], vals[i]) for i in range(len(vals)) if vals[i] is not None]
        if len(pairs) >= 2:
            h1, E1 = pairs[-2]
            h2, E2 = pairs[-1]
            if analytic_func == analytic_E_infinite:
                an = analytic_func(n, run_params.get('L',1.0))
            else:
                an = analytic_func(n)
            err1 = abs(E1 - an)
            err2 = abs(E2 - an)
            if err1 > 0 and err2 > 0:
                p = math.log(err1/err2) / math.log(h1/h2)
                print(f"n={n}: p ≈ {p:.2f}")
            else:
                print(f"n={n}: insufficient error magnitude for p estimate")
        else:
            print(f"n={n}: not enough data to estimate p")

# ---------------------------
# Grid factories
# ---------------------------
def grid_infinite_well_factory(L):
    return lambda N: ( [i*(L/N) for i in range(N+1)], L/N )

def grid_harmonic_factory(xmax):
    return lambda N: ( [ -xmax + i*(2*xmax/N) for i in range(N+1) ], 2*xmax/N )

def grid_finite_well_factory(a,b):
    return lambda N: ( [ a + i*( (b-a)/N ) for i in range(N+1) ], (b-a)/N )

# ---------------------------
# System runners
# ---------------------------
def run_infinite_well():
    L = 1.0
    print("\n*** INFINITE SQUARE WELL (L=1) ***")
    grid_func = grid_infinite_well_factory(L)
    x, h = grid_func(1000)
    V_func = lambda xi: V_infinite_well(xi, L=L)

    t0 = perf_counter()
    energies = find_eigenvalues(x, V_func, 0.0, 200.0, steps=1000)
    t1 = perf_counter()
    print(f"Found {len(energies)} eigenvalues in scan (time {t1-t0:.2f}s), first 6:")
    for i,E in enumerate(energies[:6]):
        print(f"n={i+1:2d}  E={E:.6f}  analytic={analytic_E_infinite(i+1,L):.6f}  err={abs(E-analytic_E_infinite(i+1,L)):.2e}")

    # Plot first three eigenstates
    plt.figure(figsize=(6,4))
    for i,E in enumerate(energies[:3]):
        psi, _ = numerov(E, x, V_func)
        psi = normalize(psi, h)
        plt.plot(x, psi, label=f"n={i+1}, E≈{E:.4f}")
    plt.title("Infinite Square Well: first 3 eigenstates")
    plt.xlabel("x"); plt.ylabel("ψ(x)")
    plt.legend(); plt.grid()
    save_plot("infinite_eigenstates.png")
    plt.show()

    # Ground state probability
    if energies:
        E1 = energies[0]
        psi, _ = numerov(E1, x, V_func)
        psi = normalize(psi, h)
        prob = [p*p for p in psi]
        plt.figure(figsize=(6,4))
        plt.plot(x, psi, label="ψ")
        plt.plot(x, prob, label="|ψ|^2")
        plt.title("Infinite Well: ground state ψ and |ψ|^2")
        plt.xlabel("x"); plt.legend(); plt.grid()
        save_plot("infinite_ground_prob.png")
        plt.show()

    # Convergence & error (n=1,2,3)
    run_params = {
        'grid_func': grid_infinite_well_factory(L),
        'V_func': V_func,
        'E_min': 0.0,
        'E_max': 200.0,
        'L': L,
        'steps': 800
    }
    convergence_table_for_known("Infinite Square Well", run_params, analytic_E_infinite, target_ns=[1,2,3], N_list=[200,400,800,1600])

def run_harmonic_oscillator():
    print("\n*** HARMONIC OSCILLATOR (V=1/2 x^2) ***")
    xmax = 8.0
    grid_func = grid_harmonic_factory(xmax)
    x, h = grid_func(1200)
    V_func = lambda xi: V_harmonic(xi)

    t0 = perf_counter()
    energies = find_eigenvalues(x, V_func, 0.0, 40.0, steps=1500)
    t1 = perf_counter()
    print(f"Found {len(energies)} eigenvalues in scan (time {t1-t0:.2f}s), first 6:")
    for i,E in enumerate(energies[:6]):
        print(f"n={i:2d}  E_num={E:.6f}  analytic={analytic_E_harmonic(i):.6f}  err={abs(E-analytic_E_harmonic(i)):.2e}")

    # Plot first three eigenstates
    plt.figure(figsize=(6,4))
    for i,E in enumerate(energies[:3]):
        psi, _ = numerov(E, x, V_func)
        psi = normalize(psi, h)
        plt.plot(x, psi, label=f"n={i}, E≈{E:.4f}")
    plt.title("Harmonic Oscillator: first 3 eigenstates")
    plt.xlabel("x"); plt.ylabel("ψ(x)")
    plt.legend(); plt.grid()
    save_plot("harmonic_eigenstates.png")
    plt.show()

    # Ground probability
    if energies:
        E0 = energies[0]
        psi, _ = numerov(E0, x, V_func)
        psi = normalize(psi, h)
        prob = [p*p for p in psi]
        plt.figure(figsize=(6,4))
        plt.plot(x, psi, label="ψ")
        plt.plot(x, prob, label="|ψ|^2")
        plt.title("Harmonic Oscillator: ground state ψ and |ψ|^2")
        plt.xlabel("x"); plt.legend(); plt.grid()
        save_plot("harmonic_ground_prob.png")
        plt.show()

    # Convergence & error (n=0,1,2)
    run_params = {
        'grid_func': grid_harmonic_factory(xmax),
        'V_func': V_func,
        'E_min': 0.0,
        'E_max': 40.0,
        'steps': 1200
    }
    convergence_table_for_known("Harmonic Oscillator", run_params, analytic_E_harmonic, target_ns=[0,1,2], N_list=[400,800,1200,1600])

def run_finite_well():
    print("\n*** FINITE POTENTIAL WELL (depth V0) ***")
    L = 1.0
    V0 = 50.0
    a = -1.0
    b = 2.0
    grid_func = grid_finite_well_factory(a,b)
    x, h = grid_func(1200)
    V_func = lambda xi: V_finite_well(xi, L=L, V0=V0)

    t0 = perf_counter()
    energies = find_eigenvalues(x, V_func, -V0, 0.0, steps=1200)
    t1 = perf_counter()
    print(f"Found {len(energies)} bound states (time {t1-t0:.2f}s):")
    for i,E in enumerate(energies):
        print(f"n={i+1:2d}  E={E:.6f}")

    # Plot first three bound states
    plt.figure(figsize=(6,4))
    for i,E in enumerate(energies[:3]):
        psi, _ = numerov(E, x, V_func)
        psi = normalize(psi, h)
        plt.plot(x, psi, label=f"n={i+1}, E≈{E:.4f}")
    plt.title("Finite Well: first bound eigenstates")
    plt.xlabel("x"); plt.ylabel("ψ(x)")
    plt.legend(); plt.grid()
    save_plot("finite_eigenstates.png")
    plt.show()

    # Probability density for ground bound state
    if energies:
        psi, _ = numerov(energies[0], x, V_func)
        psi = normalize(psi, h)
        prob = [p*p for p in psi]
        plt.figure(figsize=(6,4))
        plt.plot(x, psi, label="ψ")
        plt.plot(x, prob, label="|ψ|^2")
        plt.title("Finite Well: ground state ψ and |ψ|^2")
        plt.xlabel("x"); plt.legend(); plt.grid()
        save_plot("finite_ground_prob.png")
        plt.show()

    # Convergence study for first 3 bound states
    print("\nFinite well convergence (numerical only) for first 3 states")
    N_list = [300, 600, 900, 1200]
    for Ntest in N_list:
        x_test, h_test = grid_finite_well_factory(a,b)(Ntest)
        energies_test = find_eigenvalues(x_test, V_func, -V0, 0.0, steps=800)
        row = f"N={Ntest:4d}: "
        for i in range(3):
            E_val = energies_test[i] if i < len(energies_test) else None
            if E_val is None:
                row += f" n{i+1}=--- "
            else:
                row += f" n{i+1}={E_val:.6f} "
        print(row)

# ---------------------------
# Main
# ---------------------------
def main():
    t0 = perf_counter()
    run_infinite_well()
    run_harmonic_oscillator()
    run_finite_well()
    t1 = perf_counter()
    print(f"\nTotal runtime: {t1-t0:.2f} s")

if __name__ == "__main__":
    main()