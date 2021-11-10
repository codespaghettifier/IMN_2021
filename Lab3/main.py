import matplotlib.pyplot as plt

def trapeze_next(x, v, delta_t, alpha, func_f, func_g, tolerance):
    def F():
        return x1 - x - delta_t * 0.5 * (func_f(x, v) + func_f(x1, v1))

    def G():
        return v1 - v - delta_t * 0.5 * (func_g(x, v) + func_g(x1, v1))

    x1 = x
    v1 = v

    while True:
        a11 = 1
        a12 = - delta_t * 0.5
        a21 = - delta_t * 0.5 * (-2 * alpha * x1 * v1 - 1)
        a22 = 1 - delta_t * 0.5 * alpha * (1 - x1**2)

        delta_x = (-F() * a22 + G() * a12) / (a11 * a22 - a12 * a21)
        delta_v = (-a11 * G() + a21 * F()) / (a11 * a22 - a12 * a21)

        x1 += delta_x
        v1 += delta_v

        if abs(delta_x) < tolerance and abs(delta_v) < tolerance:
            break

    return x1, v1

def rk2_next(x, v, delta_t, alpha, func_f, func_g, tolerance):
    k1x = func_f(x, v)
    k1v = func_g(x, v)

    k2x = func_f(x + delta_t * k1x, v + delta_t * k1v)
    k2v = func_g(x + delta_t * k1x, v + delta_t * k1v)

    x1 = x + delta_t * 0.5 * (k1x + k2x)
    v1 = v + delta_t * 0.5 * (k1v + k2v)
    return x1, v1

def apply_numerical(x0, v0, t_min, t_max, delta_t0, alpha, func_f, func_g, func_next, func_next_tolerance, s, p, tolerance):
    def e_error(x2_delta1, x2_delta2):
        return (x2_delta1 - x2_delta2) / (2**p - 1)

    t = t_min
    delta_t  = delta_t0
    x = x0
    v = v0
    dtxv = [(0, t, x, v)]

    while t < t_max:
        x1_delta1, v1_delta1 = func_next(x, v, delta_t, alpha, func_f, func_g, func_next_tolerance)
        x2_delta1, v2_delta1 = func_next(x1_delta1, v1_delta1, delta_t, alpha, func_f, func_g, func_next_tolerance)
        x2_delta2, v2_delta2 = func_next(x, v, delta_t * 2, alpha, func_f, func_g, func_next_tolerance)

        ex = e_error(x2_delta1, x2_delta2)
        ev = e_error(v2_delta1, v2_delta2)
        e_max = abs(ex) if abs(ex) > abs(ev) else abs(ev)

        if e_max < tolerance:
            t += 2 * delta_t
            x = x2_delta1
            v = v2_delta1
            dtxv.append((delta_t, t, x, v))

        delta_t = (s * tolerance / e_max)**(1 / (p + 1)) * delta_t

    return dtxv

def main():
    def func_f(x, v):
        return  v

    def func_g(x, v):
        return alpha * (1 - x**2) * v - x

    x0 = 0.01
    v0 = 0
    delta_t0 = 1
    s = 0.75
    p = 2
    t_min = 0
    t_max = 40
    alpha = 5
    next_tolerance = 10**(-10)

    trapeze_dtxv_2 = apply_numerical(x0, v0, t_min, t_max, delta_t0, alpha, func_f, func_g, trapeze_next, next_tolerance, s, p, 10**(-2))
    trapeze_dtxv_5 = apply_numerical(x0, v0, t_min, t_max, delta_t0, alpha, func_f, func_g, trapeze_next, next_tolerance, s, p, 10**(-5))

    plt.clf()
    plt.plot([dtxv[1] for dtxv in trapeze_dtxv_2], [dtxv[2] for dtxv in trapeze_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[1] for dtxv in trapeze_dtxv_5], [dtxv[2] for dtxv in trapeze_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.savefig("trapez_x_t.png", dpi = 200)

    plt.clf()
    plt.plot([dtxv[1] for dtxv in trapeze_dtxv_2], [dtxv[3] for dtxv in trapeze_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[1] for dtxv in trapeze_dtxv_5], [dtxv[3] for dtxv in trapeze_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("v(t)")
    plt.savefig("trapez_v_t.png", dpi = 200)

    plt.clf()
    plt.plot([dtxv[1] for dtxv in trapeze_dtxv_2], [dtxv[0] for dtxv in trapeze_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[1] for dtxv in trapeze_dtxv_5], [dtxv[0] for dtxv in trapeze_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("dt(t)")
    plt.savefig("trapez_dt_t.png", dpi = 200)

    plt.clf()
    plt.plot([dtxv[2] for dtxv in trapeze_dtxv_2], [dtxv[3] for dtxv in trapeze_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[2] for dtxv in trapeze_dtxv_5], [dtxv[3] for dtxv in trapeze_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("v(x)")
    plt.savefig("trapez_v_x.png", dpi = 200)


    rk2_dtxv_2 = apply_numerical(x0, v0, t_min, t_max, delta_t0, alpha, func_f, func_g, rk2_next, next_tolerance, s, p, 10**(-2))
    rk2_dtxv_5 = apply_numerical(x0, v0, t_min, t_max, delta_t0, alpha, func_f, func_g, rk2_next, next_tolerance, s, p, 10**(-5))

    plt.clf()
    plt.plot([dtxv[1] for dtxv in rk2_dtxv_2], [dtxv[2] for dtxv in rk2_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[1] for dtxv in rk2_dtxv_5], [dtxv[2] for dtxv in rk2_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.savefig("rk2_x_t.png", dpi = 200)

    plt.clf()
    plt.plot([dtxv[1] for dtxv in rk2_dtxv_2], [dtxv[3] for dtxv in rk2_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[1] for dtxv in rk2_dtxv_5], [dtxv[3] for dtxv in rk2_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("v(t)")
    plt.savefig("rk2_v_t.png", dpi = 200)

    plt.clf()
    plt.plot([dtxv[1] for dtxv in rk2_dtxv_2], [dtxv[0] for dtxv in rk2_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[1] for dtxv in rk2_dtxv_5], [dtxv[0] for dtxv in rk2_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("dt(t)")
    plt.savefig("rk2_dt_t.png", dpi = 200)

    plt.clf()
    plt.plot([dtxv[2] for dtxv in rk2_dtxv_2], [dtxv[3] for dtxv in rk2_dtxv_2], "r-", label = "TOL = 10^(-2)", linewidth = 1)
    plt.plot([dtxv[2] for dtxv in rk2_dtxv_5], [dtxv[3] for dtxv in rk2_dtxv_5], "b-", label = "TOL = 10^(-5)", linewidth = 1)
    plt.legend(loc = "upper right")
    plt.xlabel("t")
    plt.ylabel("v(x)")
    plt.savefig("rk2_v_x.png", dpi = 200)

if __name__ == '__main__':
    main()