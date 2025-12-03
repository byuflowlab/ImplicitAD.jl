from scipy.optimize import root
from functools import partial
import jax
from jax import custom_jvp, jvp, jacfwd
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# -------- nonlinear --------
@partial(custom_jvp, nondiff_argnums=(0, 1, 3))
def implicit(solve, residual, x, p):
    return solve(x, p)

@implicit.defjvp
def implicit_jvp_jax(solve, residual, p, primals, tangents):
    x, = primals
    xdot, = tangents

    # evaluate solver
    y = solve(x, p)

    # solve for Jacobian-vector product
    _, b = jvp(lambda xtilde: residual(y, xtilde, p), (x,), (-xdot,))

    # compute partial derivatives
    A = jacfwd(lambda ytilde: residual(ytilde, x, p))(y)

    # linear solve
    ydot = jnp.linalg.solve(A, b)

    return y, ydot

# reverse case handled automatically by jax from forward definition
# -------------------------------------


# example based on OpenMDAO circuit example
# https://openmdao.org/newdocs/versions/latest/examples/circuit_analysis_examples.html

def resistor(R, Vin, Vout):
    deltaV = Vin - Vout
    I = deltaV / R
    return I

def diode(Vin, Vout, Is=1e-15, Vt=.025875):
    deltaV = Vin - Vout
    I = Is * (jnp.exp(deltaV / Vt) - 1)
    return I

def node(cin, cout):
    R = jnp.sum(cin) - jnp.sum(cout)
    return R


def circuit(Iin, Vg, R1, R2):

    def outputs(y, x, p):
        V1, V2 = y
        R1, R2 = x
        Iin, Vg = p
        r1I = resistor(R1, V1, Vg)
        r2I = resistor(R2, V1, V2)
        dI = diode(V2, Vg)
        r1 = node(jnp.array([Iin]), jnp.array([r1I, r2I]))
        r2 = node(jnp.array([r2I]), jnp.array([dI]))
        return jnp.array([r1, r2]), jnp.array([r1I, r2I, dI])

    def residuals(y, x, p):
        return outputs(y, x, p)[0]

    def solve(x, p):
        rwrap = lambda y: residuals(y, x, p)
        sol = root(rwrap, [10.0, 1], method='hybr')
        return sol.x

    x = jnp.array([R1, R2])
    p = (Iin, Vg)
    V = implicit(solve, residuals, x, p)

    _, I = outputs(V, x, p)
    return jnp.concatenate([V, I])



if __name__ == "__main__":

    from jax import vjp, jacrev

    Iin = 0.1
    Vg = 0.0
    R1 = 100.0
    R2 = 10000.0
    results = circuit(Iin, Vg, R1, R2)
    print(results)

    circuitwrapper = lambda x: circuit(Iin, Vg, x[0], x[1])
    x = jnp.array([R1, R2])
    J1 = jacfwd(circuitwrapper)(x)
    print("J1 =", J1)
    J2 = jacrev(circuitwrapper)(x)
    print("J2 =", J2)
    print("diff =", jnp.abs(J1 - J2))

    xdot = jnp.array([1.0, 0.0])
    y, ydot = jvp(circuitwrapper, (x,), (xdot,))
    print("ydot =", ydot)

    ybar = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    y, vjpfun = vjp(circuitwrapper, x)
    print("xbar =", vjpfun(ybar))


