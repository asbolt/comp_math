import numpy as np
import matplotlib.pyplot as plt

n = 10
i, j = np.indices((n, n)) + 1
A = 1.0/(i + j)
np.fill_diagonal(A, 1.0)
f = 1.0/np.arange(1, n + 1)

tol = 1e-12
maxit = 10000


def residual(x):
    return np.linalg.norm(f - A @ x)/np.linalg.norm(f)


def gauss(A, f):
    M = np.column_stack([A, f]).astype(float)
    for k in range(n):
        p = k + np.argmax(np.abs(M[k:, k]))
        M[[k, p]] = M[[p, k]]
        M[k + 1:] -= np.outer(M[k + 1:, k]/M[k, k], M[k])
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = (M[k, -1] - M[k, k + 1:n] @ x[k + 1:])/M[k, k]
    return x


def lu(A, f):
    L, U = np.eye(n), np.zeros((n, n))
    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        if abs(U[k, k]) < 1e-14:
            raise ValueError("LU неприменим: ведущий минор равен нулю")
        L[k + 1:, k] = (A[k + 1:, k] - L[k + 1:, :k] @ U[:k, k])/U[k, k]
    y = np.zeros(n)
    for k in range(n):
        y[k] = f[k] - L[k, :k] @ y[:k]
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = (y[k] - U[k, k + 1:] @ x[k + 1:])/U[k, k]
    return x


def iterate(step):
    x = np.zeros(n)
    hist = [residual(x)]
    while len(hist) <= maxit and tol <= hist[-1] < 1e10:
        x = step(x)
        hist.append(residual(x))
    return x, hist


def jacobi():
    d = np.diag(A)
    return iterate(lambda x: (f - (A - np.diag(d)) @ x)/d)


def sor(omega=1.0):
    def step(x):
        x = x.copy()
        for k in range(n):
            xk = (f[k] - A[k, :k] @ x[:k] - A[k, k + 1:] @ x[k + 1:])/A[k, k]
            x[k] += omega*(xk - x[k])
        return x
    return iterate(step)


def steepest_descent():
    def step(x):
        r = f - A @ x
        return x + (r @ r)/(A @ r @ r)*r
    return iterate(step)


def min_residual():
    def step(x):
        r = f - A @ x
        Ar = A @ r
        return x + (Ar @ r)/(Ar @ Ar)*r
    return iterate(step)


def conjugate_gradient():
    x = np.zeros(n)
    r = f.copy()
    p = r.copy()
    hist = [residual(x)]
    while len(hist) <= maxit and hist[-1] >= tol:
        Ap = A @ p
        alpha = (r @ r)/(p @ Ap)
        x = x + alpha*p
        r_new = r - alpha*Ap
        p = r_new + (r_new @ r_new)/(r @ r)*p
        r = r_new
        hist.append(residual(x))
    return x, hist


def bicgstab():
    x = np.zeros(n)
    r = f.copy()
    r0 = r.copy()
    rho = alpha = w = 1.0
    v = p = np.zeros(n)
    hist = [residual(x)]
    while len(hist) <= maxit and hist[-1] >= tol:
        rho_new = r0 @ r
        p = r + rho_new/rho*alpha/w*(p - w*v)
        v = A @ p
        alpha = rho_new/(r0 @ v)
        s = r - alpha*v
        t = A @ s
        w = (t @ s)/(t @ t)
        x = x + alpha*p + w*s
        r = s - w*t
        rho = rho_new
        hist.append(residual(x))
    return x, hist


lam = np.linalg.eigvalsh(A)
A_inv = np.column_stack([gauss(A, e) for e in np.eye(n)])
D, L, U = np.diag(np.diag(A)), np.tril(A, -1), np.triu(A, 1)
print(f"lambda_min = {lam.min():.5f}, lambda_max = {lam.max():.5f}")
for p in [1, 2, np.inf]:
    print(f"mu_{p} = {np.linalg.norm(A, p)*np.linalg.norm(A_inv, p):.4f}")
print(f"rho Якоби = {max(abs(np.linalg.eigvals(np.linalg.solve(D, L + U)))):.4f}")
print(f"rho Зейделя = {max(abs(np.linalg.eigvals(np.linalg.solve(D + L, U)))):.4f}")
print(f"Критерий останова: ||f - Ax|| / ||f|| < {tol}\n")

for name, solve in [("Гаусс", gauss), ("LU", lu)]:
    print(f"{name:<22} невязка {residual(solve(A, f)):.2e}")

omega = min(np.arange(0.05, 2, 0.05), key=lambda w: len(sor(w)[1]))
results = {
    "Якоби": jacobi(),
    "Зейдель": sor(),
    f"ВР, ω = {omega:.2f}": sor(omega),
    "Градиентный спуск": steepest_descent(),
    "Мин. невязки": min_residual(),
    "Сопр. градиенты": conjugate_gradient(),
    "BiCGStab": bicgstab(),
}

plt.figure(figsize=[9, 5])
for name, (x, hist) in results.items():
    print(f"{name:<22} невязка {hist[-1]:.2e}, итераций {len(hist) - 1}")
    plt.semilogy(hist, ".-", label=name)
plt.axhline(tol, color="k", ls=":")
plt.xlim(0, 40)
plt.ylim(1e-14, 1e3)
plt.xlabel("итерация")
plt.ylabel("||f - Ax|| / ||f||")
plt.title("Убывание невязки")
plt.grid(True, which="both", lw=0.4)
plt.legend()
plt.tight_layout()
plt.show()
