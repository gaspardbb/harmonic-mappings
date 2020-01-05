from time import time

import numpy as np
from skimage.draw import polygon, circle, rectangle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from barycenter_utils import barycentric_interpolation
from geometric_utils import plot_2D_triangulation
from surface_pre_computations import geometricMatrices, geometricQuantities, trianglesToVertices


def generate_shape(name: str, size: tuple):
    """
    Generate a shape one can use for interpolation.

    Parameters
    ----------
    name
        Shape type: "triangle", "two_circles" or "cross".
    size
        Shape size. Tuple of int.

    Returns
    -------
    Array of shape `size`.

    Raises
    ------
    ValueError
        If the name is not supported.
    """
    assert len(size) == 2

    rows, cols = size
    result = np.zeros(size)

    if name == 'triangle':
        r = np.array([0, 2, 2])
        c = np.array([1, 0, 2])
        rr, cc = polygon(r * rows // 2, c * cols // 2, shape=size)
        result[rr, cc] = 1

    elif name == 'two_circles':
        for args in [(rows // 2, cols // 8 * 2, cols // 5), (rows // 2, cols // 8 * 8, cols // 5)]:
            rr, cc = circle(*args, shape=size)
            result[rr, cc] = 1

    elif name == 'cross':
        for args in [((0, 2 * cols // 5), (rows, 3 * cols // 5)), ((2 * rows // 5, 0), (3 * rows // 5, cols))]:
            rr, cc = rectangle(*args, shape=size)
            result[rr, cc] = 1

    else:
        raise ValueError('Shape not supported.')

    result += .01
    result /= result.sum()
    return result


# @njit
def mean_of_triangles(vertices, triangles, area_triangles):
    """
    Compute the matrix used to go from triangles to vertices.

    Parameters
    ----------
    vertices: ndarray
        Matrix of the vertices. Used only to get the # of vertices.
    triangles: ndarray
        Matrix of the triangles.
    area_triangles: ndarray
        Areas of the triangles

    Returns
    -------
    A (|V|, |T|) matrix
    """
    result = np.zeros((vertices.shape[0], triangles.shape[0]))

    # for i in range(triangles.shape[0]):
    #     t = triangles[i]
    #     result[t, i] = area_triangles[t]

    for i, t in enumerate(triangles):
        result[t, i] = area_triangles[t]

    return result


class Triangulation:

    def __init__(self, vertices, edges, triangles, boundaries=None, normal=None):
        self.vertices = vertices
        self.edges = edges
        self.triangles = triangles

        self.n_vertices = vertices.shape[0]
        self.n_edges = edges.shape[0]
        self.n_triangles = triangles.shape[0]

        t_geom = time()
        print("Building geometric quantities...", end=' ')
        self.areaTriangles, self.angleTriangles, self.baseFunction = geometricQuantities(vertices, triangles, edges)
        print(f"(in {time() - t_geom:.1f}s)")

        t_mat = time()
        print("Building geometric matrices...", end=' ')
        self.gradient, self.divergence, self.laplacian = geometricMatrices(vertices, triangles, edges,
                                                                           self.areaTriangles, self.angleTriangles,
                                                                           self.baseFunction)
        print(f"(in {time() - t_mat:.1f}s)")

        t_TtoV = time()
        print("Building triangles to vertices...", end=' ')
        self.originTriangles, self.areaVertices, self.vertexTriangles = trianglesToVertices(self.vertices,
            self.triangles, self.areaTriangles)
        print(f"(in {time() - t_TtoV:.1f}s)")

        t_mean = time()
        print("Building adjacency matrix...", end=' ')
        self.mean_triangles = mean_of_triangles(vertices, triangles, self.areaTriangles)
        print(f"(in {time() - t_mean:.1f}s)")

        # Allow to sum on R3 components
        self.triangles_R3_to_R = np.kron(np.eye(self.n_triangles), np.ones((3, 1)))

        self.has_boundaries = False
        if boundaries is not None or normal is not None:
            assert boundaries is not None and normal is not None, "Boundaries and normal: provide either both or " \
                                                                  "neither"
            assert boundaries.ndim == 1 and normal.ndim == 2 and normal.shape[1] == 3
            assert boundaries.shape[0] == normal.shape[0], f"Boundaries and normal should have same length. " \
                                                           f"Got {boundaries.shape[0]}, {normal.shape[0]}."
            self.has_boundaries = True

        self.boundaries = boundaries
        self.normal = normal
        self.n_boundaries = boundaries.shape[0] if self.has_boundaries else 0

        # self.normal_product makes the product w.r.t the normal.
        if self.has_boundaries:
            k = np.kron(np.eye(self.n_boundaries), np.ones(3))
            k[k == 1] = self.normal.flatten()
            self.normal_product = k

            assert self.normal_product.shape == (self.n_boundaries, 3 * self.n_boundaries)
            assert (self.normal_product[1, 3:6] == self.normal[1]).all()

    def plot_triangulation(self, *args, **kwargs):
        plot_2D_triangulation(self.vertices, self.edges, self.triangles, self.boundaries, self.normal, *args, **kwargs)

    def plot_boundary_conditions(self, images: np.ndarray, zoom=4, ax=None):
        assert self.has_boundaries
        assert images.ndim == 3 and images.shape[0] == self.n_boundaries

        coords = [(self.vertices[self.triangles[self.boundaries[i]], :2]).mean(0) for i in range(self.n_boundaries)]
        artists = plot_at_coordinates(images, np.array(coords), zoom=zoom, ax=ax)

        return artists

    def plot_on_triangles(self, images: np.ndarray, zoom=4, ax=None):
        assert images.shape[0] == self.n_triangles

        coords = [(self.vertices[self.triangles[i], :2]).mean(0) for i in range(self.n_triangles)]
        artists = plot_at_coordinates(images, np.array(coords), zoom=zoom, ax=ax)

        return artists

    def plot_on_vertices(self, images: np.ndarray, zoom=4, ax=None):
        assert images.shape[0] == self.n_vertices

        coords = self.vertices[:, :-1]
        artists = plot_at_coordinates(images, coords, zoom=zoom, ax=ax)

        return artists


def plot_at_coordinates(images: np.ndarray, coordinates: np.ndarray, zoom: float, ax: plt.Axes=None):
    assert images.ndim == 3 and coordinates.ndim == 2
    assert images.shape[0] == coordinates.shape[0] and coordinates.shape[1] == 2
    if ax is None:
        ax = plt.gca()

    artists = []
    for (x,y), to_plot in zip(coordinates, images):
        im = OffsetImage(to_plot, zoom=zoom, cmap='coolwarm')
        box = AnnotationBbox(im, (x, y), frameon=False)
        artists.append(ax.add_artist(box))
    return artists



def triangle_boundary_condition(n: int, array_BL, array_BR, array_T, epsilon=.01, n_iter=200):
    """
    Function to get the boundary conditions for a triangle triangulation.
    Given a number of vertices per side n, 3 arrays corresponding to the corner of the triangle, computes:

    1. interpolation between bottom left/bottom right/top
    2. Fills an array corresponding to the boundary of the triangle, in the convention used so far (bottom to top):
        * First n-1 indices: the bottom row
        * Then: left then right, alternatively.

    Parameters
    ----------
    n
        Number of vertices per side.
    array_BL
        Array in the bottom left
    array_BR
        Array in the bottom right
    array_T
        Array at the top.
    epsilon
        The epsilon regularizer for the barycentric interpolation
    n_iter
        The number of iterations for the barycentric interpolation.

    Returns
    -------
    An array of size (3 * (n-2), shape_of_a_corner_array), corresponding to the boundary conditions.
    """
    assert array_BL.shape == array_BR.shape == array_T.shape, "Boundary conditions do not have the same shapes."
    shape = array_BL.shape

    bottom = barycentric_interpolation(n, array_BL, array_BR, epsilon=epsilon, n_iter=n_iter)
    right = barycentric_interpolation(n, array_BR, array_T, epsilon=epsilon, n_iter=n_iter)
    left = barycentric_interpolation(n, array_T, array_BL, epsilon=epsilon, n_iter=n_iter)

    # Result is an array of shape (#_boundary_conditions, shape_of_a_boundary_condition)
    result = np.zeros((3 * (n - 2),) + shape)

    # Filling the bottom row
    for col in range(n - 1):
        result[col] = bottom[col]

    for row in range(n - 3):
        # Filling the right
        result[n - 1 + 2 * row] = right[-2 - row]
        # Filling the left
        result[n - 1 + 2 * row + 1] = left[1 + row]

    # Filling the top
    result[-1] = array_T

    return result


if __name__ == '__main__':
    two_circles = generate_shape("two_circles", (10, 10))
    cross = generate_shape("cross", (10, 10))
    triangle = generate_shape("triangle", (10, 10))

    # r = triangle_boundary_condition(5, two_circles, cross, triangle)
    # np.save('save/boundaries_5_10.npy', r)
    r = np.load('save/boundaries_5_10.npy')

    from geometric_utils import triangle_triangulation, square_triangulation
    N_POINTS_TRIANGLE = 5
    domain = Triangulation(*triangle_triangulation(N_POINTS_TRIANGLE, get_boundary=True))

    # N_POINTS_SQUARE = 10
    # manifold = Triangulation(*square_triangulation(N_POINTS_SQUARE))

    domain.plot_triangulation(triangles_alpha=0, arrows_alpha=0, )
    # manifold.plot_triangulation()
    plt.axis('off')
    domain.plot_boundary_conditions(r, zoom=7)
    # plt.show()
