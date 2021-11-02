import math
import matplotlib.pyplot as plt

class Iterating:
    def __init__(self, delta_t, beta, n, gamma, tolerance, mi_max):
        self.delta_t = delta_t
        self.beta = beta
        self.n = n
        self.gamma = gamma
        self.tolerance = tolerance
        self.mi_max = mi_max

    def iterate(self, u):
        pass

class Pickard(Iterating):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, u):
        def next_iteration(u1):
            return u + self.delta_t / 2 * ((alpha * u - self.beta * u ** 2) + (alpha * u1 - self.beta * u1**2))

        alpha = self.beta * self.n - self.gamma
        u1 = u
        u1_mi1 = u1

        for _ in range(self.mi_max):
            u1_mi1 = next_iteration(u1)

            if abs(u1_mi1 - u1) < self.tolerance:
                break
            u1 = u1_mi1

        return u1_mi1

class Newton(Iterating):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, u):
        def next_iteration(u1):
            return u1 - (u1 - u - self.delta_t * 0.5 * ((alpha * u - self.beta * u**2) + (alpha * u1 - self.beta * u1**2))) / (1 - self.delta_t * 0.5 * (alpha - 2 * self.beta * u1))

        alpha = self.beta * self.n - self.gamma
        u1 = u
        u1_mi1 = u1

        for _ in range(self.mi_max):
            u1_mi1 = next_iteration(u1)

            if abs(u1_mi1 - u1) < self.tolerance:
                break
            u1 = u1_mi1

        return u1_mi1

def trapeze(u0, func, t_min, t_max, delta_t, iterating_object):
    xy = [(t_min, u0)]
    t = t_min
    u = u0
    while t + delta_t <= t_max:
        u1 = iterating_object.iterate(u)
        u = u + delta_t * 0.5 * (func(t, u) + func(t + delta_t, u1))
        t += delta_t
        xy.append((t, u))

    return xy

class Newton_rk2(Iterating):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self, u):
        def F1(u1p, u2p):
            return u1p - u - self.delta_t * (a11 * (alpha * u1p - self.beta * u1p**2) + a12 * (alpha * u2p - self.beta * u2p**2))

        def F2(u1p, u2p):
            return u2p - u - self.delta_t * (a21 * (alpha * u1p - self.beta * u1p**2) + a22 * (alpha * u2p - self.beta * u2p**2))

        def next_iteration(u1p, u2p):
            m11 = 1 - self.delta_t * a11 * (alpha - 2 * self.beta * u1p)
            m12 = -self.delta_t * a12 * (alpha - 2 * self.beta * u2p)
            m21 = -self.delta_t * a21 * (alpha - 2 * self.beta * u1p)
            m22 = 1 - self.delta_t * a22 * (alpha - 2 * self.beta * u2p)

            delta_u1p = (F2(u1p, u2p) * m12 - F1(u1p, u2p) * m22) / (m11 * m22 - m12 * m21)
            delta_u2p = (F1(u1p, u2p) * m21 - F2(u1p, u2p) * m11) / (m11 * m22 - m12 * m21)

            u1p = u1p + delta_u1p
            u2p = u2p + delta_u2p
            return u1p, u2p

        alpha = self.beta * self.n - self.gamma
        a11 = 0.25
        a12 = 0.25 - math.sqrt(3) / 6
        a21 = 0.25 + math.sqrt(3) / 6
        a22 = 0.25

        u1p = u
        u2p = u
        u1p_mi1 = u1p
        u2p_mi1 = u2p

        for _ in range(self.mi_max):
            u1p_mi1, u2p_mi1 = next_iteration(u1p, u2p)

            if abs(u1p_mi1 - u1p) < self.tolerance and abs(u2p_mi1 - u2p) < self.tolerance:
                break
            u1p = u1p_mi1
            u2p = u2p_mi1

        return u1p_mi1, u2p_mi1
    

def rk2(u0, func, t_min, t_max, delta_t, iterating_object):
    xy = [(t_min, u0)]
    t = t_min
    u = u0
    
    c1 = 0.5 - math.sqrt(3) / 6
    c2 = 0.5 + math.sqrt(3) / 6
    b1 = 0.5
    b2 = 0.5

    while t + delta_t <= t_max:
        u1p, u2p = iterating_object.iterate(u)
        u = u + delta_t * (b1 * func(t +  c1 * delta_t, u1p) + b2 * func(t + c2 * delta_t, u2p))
        t += delta_t
        xy.append((t, u))

    return  xy

def main():
    def func(t, u):
        return (beta * n - gamma) * u - beta * u**2

    beta = 0.001
    n = 500
    gamma = 0.1
    t_max = 100
    delta_t = 0.1
    u0 = 1
    tolerance = 10**-6
    mi_max = 20

    pickard = Pickard(delta_t, beta, n, gamma, tolerance, mi_max)
    trapeze_pickard_xy = trapeze(u0, func, 0, t_max, delta_t, pickard)
    trapeze_pickard_zy = [(xy[0], n - xy[1]) for xy in trapeze_pickard_xy]

    plt.clf()
    plt.plot([xy[0] for xy in trapeze_pickard_xy], [xy[1] for xy in trapeze_pickard_xy], "r-", label = "u(t)", linewidth = 2)
    plt.plot([xy[0] for xy in trapeze_pickard_zy], [xy[1] for xy in trapeze_pickard_zy], "r-", label = "z(t) = N - u(t)", linewidth = 1)
    plt.legend(loc = "center right")
    plt.xlabel("t")
    plt.ylabel("u(t), z(t)")
    plt.savefig("pickard.png", dpi = 200)

    newton = Newton(delta_t, beta, n, gamma, tolerance, mi_max)
    trapeze_newton_xy = trapeze(u0, func, 0, t_max, delta_t, newton)
    trapeze_newton_zy = [(xy[0], n - xy[1]) for xy in trapeze_newton_xy]

    plt.clf()
    plt.plot([xy[0] for xy in trapeze_newton_xy], [xy[1] for xy in trapeze_newton_xy], "b-", label = "u(t)", linewidth = 2)
    plt.plot([xy[0] for xy in trapeze_newton_zy], [xy[1] for xy in trapeze_newton_zy], "b-", label = "z(t) = N - u(t)", linewidth = 1)
    plt.legend(loc = "center right")
    plt.xlabel("t")
    plt.ylabel("u(t), z(t)")
    plt.savefig("newton.png", dpi = 200)

    newton_rk2 = Newton_rk2(delta_t, beta, n, gamma, tolerance, mi_max)
    rk2_xy = rk2(u0, func, 0, t_max, delta_t, newton_rk2)
    rk2_zy = [(xy[0], n - xy[1]) for xy in rk2_xy]

    plt.clf()
    plt.plot([xy[0] for xy in rk2_xy], [xy[1] for xy in rk2_xy], "k-", label = "u(t)", linewidth = 2)
    plt.plot([xy[0] for xy in rk2_zy], [xy[1] for xy in rk2_zy], "k-", label = "z(t) = N - u(t)", linewidth = 1)
    plt.legend(loc = "center right")
    plt.xlabel("t")
    plt.ylabel("u(t), z(t)")
    plt.savefig("rk2.png", dpi = 200)

if __name__ == '__main__':
    main()
