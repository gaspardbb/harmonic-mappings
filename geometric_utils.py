import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from numpy.linalg import norm


def triangle_normal(v_indices: tuple, vertices: np.ndarray, corner: bool) -> np.ndarray((3,)):
    """
    Small function to get the normal of a triangle.

    Parameters
    ----------
    v_indices
        Tuple of the indices of the vertices. Should be of length 3. *The first two indices are related to the
        external edge, through which goes the vector*. The last index completes the triangle. If corner is true,
        the last index is the corner.
    vertices
        Array of size (n_vertices, 3). Contains the coordinates of the vertices.
    corner
        Whether or not we want the normal to go through the last vertex, in case it is at a corner.

    Returns
    -------
    Numpy array of size 3, containing the coordinates of the normal.
    """
    assert len(v_indices) == 3

    if not corner:
        # The first two index are the edge toward which the normal goes
        mean = np.mean(vertices[v_indices[:2], :], axis=0)
        result = mean - vertices[v_indices[2], :]
    else:
        # The corner is the last vertex
        result = - triangle_normal(v_indices, vertices, corner=False)

    result /= norm(result)
    return result


@njit
def _square_get_edges_and_triangles_(n: int):
    """
    Utils function to parallelize the triangulation of the square.

    Parameters
    ----------
    n: int
        # of points on the square's sides

    Returns
    -------
    Edges and triangles of the triangulated square.
    """
    edges = np.zeros(((n - 1) * (3 * n - 1), 2), dtype=np.dtype('int32'))
    triangles = np.zeros((2 * (n - 1) ** 2, 3), dtype=np.dtype('int32'))

    edges_index = 0
    triangles_index = 0
    for i in range(n):
        for j in range(n):
            current = i * n + j
            right = (i + 1) * n + j
            top = (i * n) + (j + 1)
            top_right = (i + 1) * n + (j + 1)

            if i != n - 1:
                edges[edges_index] = (current, right)
                edges_index += 1
                if j != n - 1:
                    edges[edges_index] = (current, top_right)
                    edges_index += 1

            if j != n - 1:
                edges[edges_index] = (current, top)
                edges_index += 1

            if i != n - 1 and j != n - 1:
                triangles[triangles_index] = (current, right, top_right)
                triangles_index += 1
                triangles[triangles_index] = (current, top_right, top)
                triangles_index += 1
    return edges, triangles


def square_triangulation(n: int, save=False):
    """
    Return triangulation of a flat square with height 0, and size 1.
    Maybe not the fastest way to do so, but understandable.

    Parameters
    ----------
    n: int
        # of vertices on each sides
    save: bool
        Whether or not to save the result in `save/` subfolder.
    Returns
    -------
    vertices, edges, triangles
    """
    try:
        result = []
        for file in ['vertices', 'edges', 'triangles']:
            result.append(np.load('save/square_%s_%d.npy' % (file, n)))
        print("Loaded files saved in `save/` folder!")
        return result
    except FileNotFoundError:
        linspace = np.linspace(0, 1, n)
        vertices = np.stack([np.repeat(linspace, n), np.tile(linspace, n)]).T
        vertices = np.concatenate([vertices, np.zeros((n ** 2, 1))], axis=1)

        edges, triangles = _square_get_edges_and_triangles_(n)

        # assert edges_index == edges.shape[0], f"{edges_index - 1}, {edges.shape[0]}"
        # assert triangles_index == triangles.shape[0]
        assert not np.all(edges[-1] == 0)
        assert not np.all(triangles[-1] == 0)

        if save:
            np.save('save/square_vertices_%d.npy' % n, vertices)
            np.save('save/square_edges_%d.npy' % n, edges)
            np.save('save/square_triangles_%d.npy' % n, triangles)

        return vertices, edges, triangles


def triangle_triangulation(n: int, get_boundary=False):
    """
    Triangulation of an equilateral triangle.
    Most horrible thing I ever coded.
    Could be done way more efficiently.

    Parameters
    ----------
    n: int
        # of vertices on each side
    get_boundary: bool
        Whether or not to return the indices of the triangles which are part of the boundaries.
    Returns
    -------
    vertices, edges, triangles
    """
    vertices = np.zeros((n * (n+1) // 2, 3))
    v_index = 0
    x_step = 1 / (n-1)
    height = np.sqrt(3) / 2
    y_step = height / (n-1)
    for row in range(n):
        start = x_step / 2 * row
        for col in range(n-row):
            vertices[v_index] = (start + x_step * col, y_step * row, 0)
            v_index += 1

    dist = pairwise_dist(vertices)

    edges = np.argwhere(np.logical_and(dist <= x_step+1e-7, dist != 0))
    edges = edges[edges[:, 0] > edges[:, 1]]
    assert edges.shape[0] == 3 * (n-1)*n//2

    if get_boundary:
        # Retain the indices of the boundaries
        boundaries = []
        # Retain the normal vectors to the boundaries
        normal = []

    triangles = np.zeros(((n-1)**2, 3), dtype=int)
    t_index = 0
    v_index = 0
    for row in range(n):
        for col in range(n-row):
            if col != n-row-1 and row != n-1:
                triangles[t_index] = (v_index, v_index+1, v_index + n - row)
                t_index += 1
            if col != 0 and col != n-row-1:
                triangles[t_index] = (v_index, v_index + n - row, v_index + n - row - 1)
                t_index += 1

            if get_boundary:
                # Special case for the corner triangles
                if row == 0 and col == 0:  # bottom left corner
                    boundaries.append(t_index-1)
                    # print('BL')
                    normal.append(triangle_normal((1, n, 0), vertices, corner=True))
                elif row == 0 and col == n-row-2:  # bottom right corner
                    boundaries.append(t_index-2)
                    # print('BR')
                    normal.append(triangle_normal((n - 2, 2 * n - 2, n - 1), vertices, corner=True))
                elif row == n-1:  # top
                    boundaries.append(t_index-1)
                    # print('T')
                    last_idx = n * (n+1) // 2 - 1
                    normal.append(triangle_normal((last_idx - 1, last_idx - 2, last_idx), vertices, corner=True))
                else:
                    if row == 0 and col < n-row-2:  # Bottom row
                        boundaries.append(t_index-2)
                        # print('bottom')
                        normal.append(triangle_normal((v_index, v_index + 1, v_index + n - row), vertices, False))
                    elif col == 0 and row < n-2:  # Left side
                        boundaries.append(t_index-1)
                        # print('left')
                        normal.append(triangle_normal((v_index, v_index + n - row, v_index + 1), vertices, False))
                    elif col == n-row-1 and 0 < row < n-2:  # Right side
                        boundaries.append(t_index-2)
                        # print('right')
                        # print(v_index+n-row)
                        normal.append(triangle_normal((v_index + n - row - 1, v_index, v_index - 1), vertices, False))
            v_index += 1

    if get_boundary:
        # Ok. That's ugly. Not scalable. But it works for small n.
        boundaries = np.array(boundaries)
        normal = np.array(normal)

        assert boundaries.shape[0] == normal.shape[0]
        return vertices, edges, triangles, boundaries, normal

    return vertices, edges, triangles


def plot_2D_triangulation(vertices=None, edges=None, triangles=None, boundaries=None, normal=None, label=True,
                          ax=None,
                          triangles_alpha: float=1,
                          arrows_alpha: float= 1):
    """
    Plot the two first components of a triangulation.

    Parameters
    ----------
    vertices:
        array of the vertices: (n_samples, n_features=2)
    edges
        array of the edges: (n_edges, 2) : (id1, id2)
    triangles
        array of the triangles: (n_triangles, 3): (id1, id2, id3)
    label
        Wether to show the label of each vertex
    ax
        To plot on a specific ax.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if edges is not None:
        for v_ind in edges:
            ax.plot(vertices[v_ind, 0], vertices[v_ind, 1], "k-")
    if triangles is not None:
        for v_ind in triangles:
            ax.fill(vertices[v_ind, 0], vertices[v_ind, 1], alpha=triangles_alpha)
    if vertices is not None:
        ax.scatter(vertices[:, 0], vertices[:, 1])
    if label:
        for i, v in enumerate(vertices):
            ax.text(v[0], v[1], i)
    if boundaries is not None:
        for b in boundaries:
            origin_t = np.mean(vertices[triangles[b], :], axis=0)
            ax.text(origin_t[0], origin_t[1], b, fontweight='bold')
            ax.plot(origin_t[0], origin_t[1], "r*")
    if normal is not None:
        assert normal.shape[0] == boundaries.shape[0]
        for b, n in zip(boundaries, normal):
            origin_t = np.mean(vertices[triangles[b], :], axis=0)
            ax.arrow(origin_t[0], origin_t[1], n[0]/6, n[1]/6, width=.01, alpha=arrows_alpha)
    ax.set_aspect('equal')


def pairwise_dist(x: np.ndarray):
    return np.linalg.norm(x[..., np.newaxis] - x.T, axis=1)


if __name__ == '__main__':
    # Check square
    # v, e, t = square_triangulation(30, save=True)
    # plot_2D_triangulation(v, e, t)

    # Check triangle
    v, e, t, b, n = triangle_triangulation(5, get_boundary=True)
    plot_2D_triangulation(v, e, t, b, n)
    print(b)
    print(n)
    plt.show()