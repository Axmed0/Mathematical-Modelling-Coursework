#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:10:54 2025

@author: ahmed
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Q1 forcing term: the “unobtainium machine” turns on briefly every 0.6 s
# ---------------------------------------------------------------------
def f(x, t):
    """
    Parameters
    ----------
    x : float
        Spatial coordinate in metres.
    t : float
        Time in seconds.

    Returns
    -------
    float
        The machine’s output f(x,t).

        • The cycle length is T = 0.6 s.
        • During each cycle it is ON from 0.05 s to 0.10 s and OFF otherwise.
        • While ON the spatial profile is a triangle:
              - Base goes from x = 0.5 m to 1.3 m
              - Peak value 10 at x = 0.9 m
    """
    # Figure out “where” we are inside the 0.6 s cycle
    T = 0.6
    cycle_t = t % T
    if cycle_t < 0.05 or cycle_t > 0.10:
        return 0.0  # machine is off → no injection

    # Build the triangular spatial profile while the machine is ON
    x_peak     = 0.9
    half_width = 0.4
    left, right = x_peak - half_width, x_peak + half_width
    if x <= left or x >= right:
        return 0.0  # x is outside the triangle’s base

    slope = 10.0 / half_width                      # rise/run of the triangle
    return slope * (x - left) if x <= x_peak else slope * (right - x)


# ---------------------------------------------------------------------
# 2‑D composite Simpson’s rule for ∫∫ f(x,t) dx dt
# ---------------------------------------------------------------------
def calculate_double_simpsons_integral(Tfinal, f, Nx, Nt):
    """
    Approximate the double integral


    using composite Simpson’s rule in both x and t.

    Parameters
    ----------
    Tfinal : float
        Upper limit of the time integral.
    f : callable
        Function of two variables, f(x,t).
    Nx : int
        Number of sub‑intervals in x (must be even).
    Nt : int
        Number of sub‑intervals in t (must be even).

    Returns
    -------
    float
        Simpson approximation to the double integral.
    """
    # Simpson’s rule needs an even number of panels in each direction
    if Nx % 2 != 0:
        raise RuntimeError("Nx must be even for Simpson’s rule")
    if Nt % 2 != 0:
        raise RuntimeError("Nt must be even for Simpson’s rule")

    # Grid spacing in space (x) and time (t)
    x0, x1 = 0.0, 3.6
    t0, t1 = 0.0, Tfinal
    hx = (x1 - x0) / Nx
    ht = (t1 - t0) / Nt

    # Grid nodes
    xs = np.linspace(x0, x1, Nx + 1)
    ts = np.linspace(t0, t1, Nt + 1)

    total = 0.0
    # Double loop: outer loop over time, inner loop over space
    for j, tj in enumerate(ts):
        w_t = 1 if j in (0, Nt) else (4 if j % 2 else 2)  # Simpson weight in t
        for i, xi in enumerate(xs):
            w_x = 1 if i in (0, Nx) else (4 if i % 2 else 2)  # Simpson weight in x
            total += w_x * w_t * f(xi, tj)  # accumulate weighted f(xi,tj)

    # Combine the spacings and Simpson factor (1/9) for the final answer
    return (hx * ht / 9.0) * total



# # ---------------------------------------------------------------------
# # ONLY THE CODE IN THE TWO FUNCTIONS ABOVE WILL BE TESTED.
# # Everything below can be commented out or deleted for submission.
# # ---------------------------------------------------------------------

# # Vectorise f so we can call it on NumPy arrays
# vectorized_f = np.vectorize(f)

# # Quick plot of f(x, t_sample) just to check the triangle looks right
# my_integral_top_limit = 3.6          # x‑domain length
# x_plot = np.linspace(0, my_integral_top_limit, 1001)
# t_sample = 0.07                      # inside the ON window
# plt.plot(x_plot, vectorized_f(x_plot, t_sample))
# plt.xlabel("x (m)")
# plt.ylabel(f"f(x, t = {t_sample:.2f} s)")
# plt.title("Forcing profile while the machine is ON")
# plt.grid(True)
# plt.show()

# # Simpson test: integrate over one full 0.6 s cycle
# Tfinal = 0.6
# Nx = 360
# Nt = 600
# print("∬ f(x,t) dx dt over one cycle =",
#       calculate_double_simpsons_integral(Tfinal, f, Nx, Nt), "\n")
