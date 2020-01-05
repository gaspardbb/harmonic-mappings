# Dynamical OT on discrete surfaces

Review of an article by Lavenant et Al. [1]. Focus on harmonic mappings.

## Files

In order to work, need `surface_pre_computations.py` from the original github repository: https://github.com/HugoLav/DynamicalOTSurfaces

* `geometric_utils.py`: contain some tools to build the triangulation of a square or a triangle
* `harmonic_utils.py`: contain some tools to prepare the triangulations with the right boundaries
* `harmonic.py`: the optimization of the dual problem, in CVXPY (Disciplined convex programming, no use of the SOC structure)

## Issues

The output of the solver is unbounded; not usable, for now.

## References

[1] *Dynamical Optimal Transport on Discrete Surfaces*
Hugo Lavenant (LM-Orsay), Sebastian Claici (MIT), Edward Chien (MIT), Justin Solomon (MIT)

(Submitted on 19 Sep 2018)

We propose a technique for interpolating between probability distributions on discrete surfaces, based on the theory of optimal transport. Unlike previous attempts that use linear programming, our method is based on a dynamical formulation of quadratic optimal transport proposed for flat domains by Benamou and Brenier [2000], adapted to discrete surfaces. Our structure-preserving construction yields a Riemannian metric on the (finite-dimensional) space of probability distributions on a discrete surface, which translates the so-called Otto calculus to discrete language. From a practical perspective, our technique provides a smooth interpolation between distributions on discrete surfaces with less diffusion than state-of-the-art algorithms involving entropic regularization. Beyond interpolation, we show how our discrete notion of optimal transport extends to other tasks, such as distribution-valued Dirichlet problems and time integration of gradient flows.