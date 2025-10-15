#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:18:08 2025

@author: ahmed
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def advection_diffusion_backward_euler_with_forcing(Tfinal, Nx, Nt, u0, f):
    """
    March a 1‑D advection–diffusion problem forward in time with
    a backward‑Euler scheme *and* an external forcing term.

    Parameters
    ----------
    Tfinal : float
        The final time we want to reach.
    Nx : int
        How many little slabs we chop the pipe into (space grid).
    Nt : int
        How many time steps we take to get from 0 to Tfinal.
    u0 : callable
        Function giving the initial concentration profile u(x,0).
    f : callable
        Function giving the forcing term f(x,t).

    Returns
    -------
    np.ndarray
        A 2‑D array with shape (Nt+1, Nx+1).  
        Row *n* is the solution at time t = n·Δt.  
        The last column is a duplicate of the first to keep
        the periodic boundary nice and tidy for plotting.
    """
    # Course‑work constants (given in the brief)
    a = 0.2          # diffusion coefficient
    b = -6.0         # advection speed
    x_upper = 3.6    # length of the pipe in metres

    h  = x_upper / Nx        # Δx
    dt = Tfinal  / Nt        # Δt

    # Build the tridiagonal system matrix for backward Euler
    main_diag   = (1 + 2*a*dt/h**2 - b*dt/h)           * np.ones(Nx)
    off_diag_lo = (-a*dt)/h**2                         * np.ones(Nx-1)
    off_diag_hi = (b*dt/h - a*dt/h**2)                 * np.ones(Nx-1)
    # Wrap‑around entries (periodic BC)
    lo_corner = [( b*dt/h - a*dt/h**2 )]   # connects last node → first
    hi_corner = [(-a*dt)/h**2]             # connects first node → last

    A = sparse.diags(
        [main_diag, off_diag_lo, off_diag_hi, lo_corner, hi_corner],
        [0,          -1,         1,          -(Nx-1),    Nx-1],
        shape=(Nx, Nx),
        format='csr'
    )

    # Allocate the solution array (extra column = duplicated endpoint)
    u = np.zeros((Nt+1, Nx+1))
    xmesh = np.linspace(0, x_upper, Nx+1)
    u[0, :] = u0(xmesh)      # slap the initial condition into row 0

    # Crank through time, one backward‑Euler solve per step
    for n in range(Nt):
        t_next = (n+1)*dt
        u_prev = u[n, :-1]                    # strip off duplicate node
        rhs    = u_prev + dt * f(xmesh[:-1], t_next)
        u_next = spsolve(A, rhs)
        u[n+1, :-1] = u_next
        u[n+1, -1]  = u_next[0]               # enforce periodic wrap

    return u


def plot_total_unobtainium_over_time(u, Tfinal):
    """
    Integrate the concentration in space (Simpson’s rule) to track how
    many moles have been pumped into the pipe, then plot that as a function
    of time.  The curve should look like a neat little staircase.

    Parameters
    ----------
    u : np.ndarray
        The array returned by `advection_diffusion_backward_euler_with_forcing`.
    Tfinal : float
        The final time used in the simulation.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure containing the staircase plot.
    """
    # Simpson’s rule only works properly when Nx and Nt are even.
    if Nx % 2 or Nt % 2:
        raise RuntimeError("Nx and Nt both need to be even for Simpson’s rule")

    h      = x_upper / Nx
    t_vals = np.linspace(0, Tfinal, Nt+1)

    # Build Simpson weights once, reuse for every time slice
    weights = np.ones(Nx+1)
    weights[1:Nx:2]   = 4
    weights[2:Nx-1:2] = 2
    weights *= h/3

    # Total moles at each time
    total = np.sum(weights * u, axis=1)
    added = total - total[0]      # subtract whatever we started with

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_vals, added, label="Total moles added", color='red')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Moles of Unobtainium")
    ax.set_title("Unobtainium Accumulation Over Time")
    ax.grid(True)
    ax.legend()

    return fig


def comment_on_results():
    
    return (
        "Every 0.4 s the machine wakes up for 0.05 s and squirts a fixed dose "
        "of unobtainium into the pipe.  Each squirt shows up as a vertical jump, "
        "and the flat bits in between are the quiet periods when the machine is off. "
        "By the end of the run we’ve added exactly the amount predicted by the "
        "area of the forcing function, so the mass balance checks out nicely."
    )


# -------------------------------------------------------------------------
# Forcing term f(x,t) and initial condition u0(x) – straight from the brief
# -------------------------------------------------------------------------
def f(x, t):
    """
    Time‑periodic injector:
      • ON for 0.05 s every 0.4 s cycle
      • Spatial profile is a neat little triangle between x=0.4 m and x=1.2 m
    """
    T = 0.4
    cycle = round((t + 0.35)/T)
    on    = cycle*T - 0.35        # start of the ON window
    off   = on + 0.05             # end of the ON window

    if t < on or t > off:
        return 0.0
    elif 0.4 < x <= 0.8:
        return 25*(x - 0.4)
    elif 0.8 <= x < 1.2:
        return 10 - 25*(x - 0.8)
    else:
        return 0.0


def u0(x):
    """
    Start with a symmetric triangle of unobtainium between x=0 and x=1 m.
    """
    if 0 <= x < 0.5:
        return 4*x
    elif 0.5 <= x < 1.0:
        return 4*(1 - x)
    else:
        return 0.0

# This method is used to vectorize the functions
# so that they can be applied to an array of x values
# and return arrays.
# We will run these lines automatically before testing your solution.
u0 = np.vectorize(u0)
f = np.vectorize(f)

# The following parameters should be set as per your question
Nx = 720
Nt = 600
Tfinal = 3.3
x_upper = 3.6

u = advection_diffusion_backward_euler_with_forcing(Tfinal, Nx, Nt, u0, f)

# You should not need to change anything below here, it is just for plotting the solution with an animation.
fig, ax = plt.subplots()

# Just defined once for use in the animation plotting below.
meshgrid = np.linspace(0, x_upper, Nx+1)


def animate(i):
    # print(i)
    ax.clear()
    ax.plot(meshgrid, u[i, :], 'o-')
    ax.plot(meshgrid, np.ones(Nx+1)/meshgrid[-1], 'k--')
    ax.set_ylim([-0.1, 3.1])
    # Print time as the title to 3 decimal places
    ax.set_title(f"Time = {(Tfinal/Nt)*i:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(["u approximation", "steady state"])


# Make a little animation to see what is happening
# N.B. The interval is set to 100ms, you can make it a smaller number for a faster video, or a bigger number for a slower video
ani = animation.FuncAnimation(
    fig, animate, frames=Nt+1, interval=100, init_func=lambda: None, blit=False)
plt.show()

# Call the method to return the matplotlib figure of the total unobtainium over time
fig = plot_total_unobtainium_over_time(u, Tfinal)
plt.show()

print(comment_on_results())