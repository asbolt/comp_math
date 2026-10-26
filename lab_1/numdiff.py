import numpy as np
import matplotlib.pyplot as plt

x0 = 1.0
h = 2.0 / 2.0 ** np.arange(1, 22)

functions = [
    ("sin(x^2)", lambda x: np.sin(x**2), lambda x: 2*x*np.cos(x**2)),
    ("cos(sin(x))", lambda x: np.cos(np.sin(x)), lambda x: -np.sin(np.sin(x))*np.cos(x)),
    ("exp(sin(cos(x)))", lambda x: np.exp(np.sin(np.cos(x))),
     lambda x: -np.exp(np.sin(np.cos(x)))*np.cos(np.cos(x))*np.sin(x)),
    ("ln(x+3)", lambda x: np.log(x + 3), lambda x: 1/(x + 3)),
    ("(x+3)^0.5", lambda x: np.sqrt(x + 3), lambda x: 0.5/np.sqrt(x + 3)),
]

methods = [
    lambda f, x, h: (f(x + h) - f(x))/h,
    lambda f, x, h: (f(x) - f(x - h))/h,
    lambda f, x, h: (f(x + h) - f(x - h))/(2*h),
    lambda f, x, h: 4/3*(f(x + h) - f(x - h))/(2*h) - 1/3*(f(x + 2*h) - f(x - 2*h))/(4*h),
    lambda f, x, h: 3/2*(f(x + h) - f(x - h))/(2*h) - 3/5*(f(x + 2*h) - f(x - 2*h))/(4*h)
    + 1/10*(f(x + 3*h) - f(x - 3*h))/(6*h),
]

for name, f, df in functions:
    plt.figure(figsize=[8, 5])
    for i, method in enumerate(methods, start=1):
        err = np.abs(method(f, x0, h) - df(x0))
        plt.loglog(h, err, "o-", ms=4, label=f"формула {i}")
    plt.title(f"Погрешность производной {name} в точке x = {x0}")
    plt.xlabel("h")
    plt.ylabel("абсолютная погрешность")
    plt.grid(True, which="both", lw=0.4)
    plt.legend()
    plt.tight_layout()

plt.show()
