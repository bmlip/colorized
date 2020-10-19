using SpecialFunctions: gamma
using PyCall
np = pyimport("numpy")
plt = pyimport("matplotlib.pyplot")
tri = pyimport("matplotlib.tri")


function tri_area(xy, pair)
    """The area of the triangle formed by point xy and another pair or points."""
    return 0.5 * np.linalg.norm(np.cross(pair[1,:] .- xy, pair[2,:] .- xy))
end

function xy2bc(xy, pair_corners; tol=1.e-4, AREA=0.5*1*0.75^0.5)
    """Converts 2D Cartesian coordinates to barycentric."""
    coords = np.array([tri_area(xy, p) for p in pair_corners]) / AREA
    return np.clip(coords, tol, 1.0 - tol)
end

function pdf_Dir(θ, α)
    """Plots pdf of Dirichlet distribution"""
    return prod(gamma.(α))/gamma(prod(α)) * prod(θ.^α)
end

function pdf_contours_simplex(concentration_params; nlevels=200, subdiv=8)
    """Generate pdf contours."""
    
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75^0.5]])
    AREA = 0.5 * 1 * 0.75^0.5
    triangle = tri.Triangulation(corners[:, 1], corners[:, 2])

    pair_corners = [corners[np.roll(range(1, length=3), -i)[2:end],:] for i in range(0, length=3)]

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [pdf_Dir(xy2bc(xy, pair_corners, AREA=AREA), concentration_params) for xy in zip(trimesh.x, trimesh.y)]

    return trimesh, pvals
    # plt.tricontourf(trimesh, pvals, nlevels, cmap="jet")
    # plt.axis("equal")
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.75^0.5)
    # plt.axis("off")
    # plt.show()
end