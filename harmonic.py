from harmonic_utils import Triangulation, triangle_boundary_condition, generate_shape
from geometric_utils import square_triangulation, triangle_triangulation
import cvxpy as cp
import numpy as np

N_POINTS_SQUARE = 10
N_POINTS_TRIANGLE = 5

# Manifold
manifold = Triangulation(*square_triangulation(N_POINTS_SQUARE))

# Domain
domain = Triangulation(*triangle_triangulation(N_POINTS_TRIANGLE, get_boundary=True))

# Printing some information
print("Vertices, Edges, Triangles :\n"
      f"Domain: {domain.n_vertices}, {domain.n_edges}, {domain.n_triangles}\n"
      f"Manifold: {manifold.n_vertices}, {manifold.n_edges}, {manifold.n_triangles}")


# BOUNDARY CONDITION

# two_circles = generate_shape("two_circles", (N_POINTS_SQUARE, N_POINTS_SQUARE))
# cross = generate_shape("cross", (N_POINTS_SQUARE, N_POINTS_SQUARE))
# triangle = generate_shape("triangle", (N_POINTS_SQUARE, N_POINTS_SQUARE))
#
# boundaries = triangle_boundary_condition(N_POINTS_TRIANGLE, two_circles, cross, triangle)
#
boundaries = np.load('save/boundaries_5_10.npy')
#
# boundaries = np.flip(boundaries, axis=0)

n_boundaries, n_rows, n_cols = boundaries.shape
assert n_boundaries == domain.n_boundaries
assert n_rows * n_cols == manifold.n_vertices

boundaries = boundaries.reshape(n_boundaries, n_rows * n_cols)

# Adding some processing on the boundaries
assert boundaries.min() > -1e-10
boundaries[boundaries < 0] = 0  # Not sure if that changes anything
boundaries = boundaries / boundaries.sum(axis=1)[:, np.newaxis]

# VARIABLE

phi = cp.Variable((3 * domain.n_triangles, manifold.n_vertices), name='phi')
print(f"Primal variable shape: {phi.shape}")

# OBJECTIVE

objective = phi

# Slicing to get the boundary term
indices = ((domain.boundaries * 3)[:, np.newaxis] + np.arange(3)).flatten()
objective = objective[indices]
assert objective.shape == (3 * domain.n_boundaries, manifold.n_vertices)

# Making the product with the outer normal
objective = domain.normal_product @ objective
assert objective.shape == (domain.n_boundaries, manifold.n_vertices)

# Multiplying with the measure at the boundary
objective = cp.multiply(objective, boundaries)
objective = cp.multiply(objective, np.ones((domain.n_boundaries, 1)) * manifold.areaVertices)
assert objective.shape == (domain.n_boundaries, manifold.n_vertices)

# Taking the sum over the vertices of the manifold
objective = cp.sum(objective, axis=1)

# Multiplying by the area of the boundary (the dx measure) and summing (ie. integrating)
objective = cp.multiply(objective, domain.areaTriangles[domain.boundaries])
objective = cp.sum(objective, axis=0)

# CONSTRAINTS

# Divergence of domain
divergence = domain.divergence @ phi
assert divergence.shape == (domain.n_vertices, manifold.n_vertices)

# Computing the gradient
gradient = manifold.gradient @ phi.T

# And then its norm on R^(3*3)
gradient_norm = cp.square(gradient)
gradient_norm = manifold.triangles_R3_to_R.T @ gradient_norm @ domain.triangles_R3_to_R
assert gradient_norm.shape == (manifold.n_triangles, domain.n_triangles)

# Averaging to put it on the vertices grid
gradient_norm = manifold.mean_triangles @ gradient_norm @ domain.mean_triangles.T

# Dividing by the vertices dual cell area
gradient_norm = cp.multiply(gradient_norm,
                            1/(manifold.areaVertices[:, np.newaxis] * domain.areaVertices[np.newaxis,:]))
gradient_norm = gradient_norm.T / 2
assert gradient_norm.shape == (domain.n_vertices, manifold.n_vertices)

# Definition of the dual variables
A = divergence
B = gradient

constraint_dual_A = cp.Zero(A - divergence)
constraint_dual_B = cp.Zero(B - gradient)

# constraints = [constraint_dual_A, A + gradient_norm <= 0, cp.norm(objective) <= 1]
# constraints = [divergence + gradient_norm <= 0, cp.norm(objective) <= 1]
constraints = [constraint_dual_A, constraint_dual_B, divergence + gradient_norm <= 0, cp.norm(objective) <= 1e3]

obj = cp.Maximize(objective)

prob = cp.Problem(obj, constraints)

prob.solve(verbose=True)


# Debug
import matplotlib.pyplot as plt

result = [phi.value]
result.append(result[-1][indices])
result.append(domain.normal_product @ result[-1])
result.append(result[-1] * boundaries * np.ones((domain.n_boundaries, 1)) * manifold.areaVertices)

fig, ax = plt.subplots()
mu = A.value
mu = mu.reshape(15, 10, 10)
domain.plot_triangulation(arrows_alpha=0,
                          triangles_alpha=0, ax=ax)
domain.plot_on_vertices(mu, ax=ax, zoom=7)
ax.axis('off')

fig2, ax = plt.subplots()
final_phi = result[3].reshape(9, 10, 10)
domain.plot_triangulation(arrows_alpha=0,
                          triangles_alpha=0, ax=ax)
domain.plot_boundary_conditions(final_phi, ax=ax, zoom=7)
ax.axis('off')

#
# fig3, axes = plt.subplots(1, 4)
# for r, ax in zip(result, axes):
#     ax.imshow(r)
#     ax.axis('off')

# fig, axes = plt.subplots(n_boundaries)
# for i, (b,ax) in enumerate(zip(result[-1], axes.flatten())):
#     ax.imshow(b.reshape(10, 10))
#     ax.set_title(i)

plt.show()