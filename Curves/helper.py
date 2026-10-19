import numpy as np


def plot_spline_2d(spline, *, ax, num_points=400, **plot_kwargs):
    """Plot a 2D parametric spline on a matplotlib axis."""
    t = np.linspace(spline.grid[0], spline.grid[-1], num_points)
    points = spline.evaluate(t)
    return ax.plot(points[:, 0], points[:, 1], **plot_kwargs)


def plot_tangent_2d(spline, t, *, ax, scale=1.0, **quiver_kwargs):
    """Plot the tangent vector of a 2D spline at parameter value t."""
    point = spline.evaluate(t)
    tangent = spline.evaluate(t, n=1)
    return ax.quiver(
        point[0],
        point[1],
        tangent[0],
        tangent[1],
        angles="xy",
        scale_units="xy",
        scale=1 / scale if scale != 0 else 1.0,
        **quiver_kwargs,
    )
