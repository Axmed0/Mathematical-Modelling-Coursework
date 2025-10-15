#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:17:27 2025

@author: ahmed
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
# Use an interactive backend so the plot window appears on screen
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math

# ----------------------------------------------------------------------------
# A little solver for the unforced 1‑D advection–diffusion equation
#
#     u_t − a·u_xx + b·u_x = 0         on   0 ≤ x ≤ L ,  t > 0
#
# with *periodic* boundary conditions.  We step in time with the
# backward‑Euler (implicit) method and finite differences in space.
# ----------------------------------------------------------------------------
def advection_diffusion_backward_euler(Tfinal, Nx, Nt, u0):
    """
    March the PDE forward to *Tfinal* using backward Euler.

    Parameters
    ----------
    Tfinal : float
        Time at which we want the solution.
    Nx : int
        Number of spatial intervals (so Nx+1 grid points).
    Nt : int
        Number of time steps between 0 and Tfinal.
    u0 : callable
        A Python function giving the initial profile u(x,0).

    Returns
    -------
    u : ndarray, shape (Nt+1, Nx+1)
        Row *n* contains the numerical solution at time t = n·Δt.
        The last column duplicates the first so the periodic BC is
        explicit in the array (handy for plotting).
    """
    # Geometry and grid spacing
    L  = x_upper       # pipe length (pulled from a global)
    dx = L / Nx
    dt = Tfinal / Nt

    # Physical parameters (from the coursework sheet)
    a = 0.15           # diffusion coefficient
    b = -10.0          # advection speed

    Np = Nx + 1        # total number of grid points (including x = L)

    # ---------------------------------------------------------------------
    # Build sparse matrices for the second derivative (D2) and the
    # first derivative (D1) with an upwind bias that matches the sign of *b*.
    # ---------------------------------------------------------------------
    D2 = sparse.lil_matrix((Np, Np))
    D1 = sparse.lil_matrix((Np, Np))
    for i in range(Np):
        im = (i - 1) % Np          # left neighbour  (periodic wrap)
        ip = (i + 1) % Np          # right neighbour (periodic wrap)

        # Classic second‑order central stencil for diffusion
        D2[i, im] =  1.0 / dx**2
        D2[i, i]  = -2.0 / dx**2
        D2[i, ip] =  1.0 / dx**2

        # Upwind stencil for advection
        if b >= 0:
            D1[i, im] = -1.0 / dx
            D1[i, i]   =  1.0 / dx
        else:
            D1[i, i]   = -1.0 / dx
            D1[i, ip]  =  1.0 / dx

    # Backward‑Euler system:  (I − dt·(a·D2 − b·D1)) · u^{n+1} = u^n
    A = a * D2 - b * D1
    M = sparse.eye(Np) - dt * A
    M = M.tocsr()      # use CSR for fast solves

    # ---------------------------------------------------------------------
    # Set up the solution array and drop in the initial condition
    # ---------------------------------------------------------------------
    u = np.zeros((Nt+1, Np))
    x = np.linspace(0, L, Np)
    u[0, :] = u0(x)

    # Time‑stepping loop
    for n in range(Nt):
        u[n+1, :] = spsolve(M, u[n, :])

    return u


# ----------------------------------------------------------------------------
# Initial condition helper
# ----------------------------------------------------------------------------
def u0(x):
    """
   triangular pulse:

        4 x          for 0 ≤ x < 0.5
        4 (1 − x)    for 0.5 ≤ x < 1
        0            everywhere else
    """
    return np.where(
        x < 0.5,
        4.0 * x,
        np.where(
            x < 1.0,
            4.0 * (1.0 - x),
            0.0
        )
    )


# ----------------------------------------------------------------------------
# Main: set up parameters, run solver, and animate
# ----------------------------------------------------------------------------
# Vectorize initial condition
u0 = np.vectorize(u0)

# Parameters
Nx      = 720     # spatial subdivisions
Nt      = 600     # time steps
Tfinal  = 4.0     # final time
x_upper = 3.6     # domain length L

# Compute solution
u = advection_diffusion_backward_euler(Tfinal, Nx, Nt, u0)
# Spatial grid for plotting
meshgrid = np.linspace(0, x_upper, Nx+1)

# Compute steady-state value (mass conservation)
mass = np.trapz(u[0, :], meshgrid)
steady = mass / x_upper

# Animated evolution via simple loop + pause
plt.ion()
fig, ax = plt.subplots()
for n in range(Nt+1):
    ax.clear()
    ax.plot(meshgrid, u[n, :], 'o-')
    ax.plot(meshgrid, np.ones_like(meshgrid) * steady, 'k--')
    ax.set_ylim([-0.1, 3.1])
    ax.set_title(f"Time = {n * Tfinal/Nt:.3f}")
    plt.pause(0.05)


