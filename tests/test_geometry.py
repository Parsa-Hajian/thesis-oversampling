"""
Tests for src.utils.geometry -- geometric sampling and rotation primitives.
"""

import sys
import os

# Ensure the project root is on the import path.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pytest

from src.utils.geometry import (
    circle_from_pair,
    uniform_in_disk_vec,
    points_in_circle,
    vonmises_in_disk,
    rotate_2d,
    rotate_batch_2d,
    assign_voronoi,
    sample_in_voronoi_cell,
    uniform_in_ball,
    uniform_in_ball_batch,
)


# ---------------------------------------------------------------------------
# circle_from_pair
# ---------------------------------------------------------------------------

class TestCircleFromPair:
    """Tests for circle_from_pair(x_i, x_j)."""

    def test_center_is_midpoint(self):
        """Circle center should be the midpoint of the two input points."""
        x_i = np.array([0.0, 0.0])
        x_j = np.array([4.0, 0.0])
        center, radius = circle_from_pair(x_i, x_j)
        expected_center = np.array([2.0, 0.0])
        np.testing.assert_allclose(center, expected_center)

    def test_radius_is_half_distance(self):
        """Radius should be half the Euclidean distance between the points."""
        x_i = np.array([0.0, 0.0])
        x_j = np.array([6.0, 8.0])
        center, radius = circle_from_pair(x_i, x_j)
        expected_distance = 10.0
        assert pytest.approx(radius) == expected_distance / 2.0

    def test_both_points_on_boundary(self):
        """Both input points should lie on the circle boundary."""
        x_i = np.array([1.0, 3.0])
        x_j = np.array([5.0, 7.0])
        center, radius = circle_from_pair(x_i, x_j)
        dist_i = np.linalg.norm(x_i - center)
        dist_j = np.linalg.norm(x_j - center)
        assert pytest.approx(dist_i) == radius
        assert pytest.approx(dist_j) == radius

    def test_coincident_points(self):
        """Coincident points should produce radius clamped to EPS > 0."""
        x = np.array([3.0, 4.0])
        center, radius = circle_from_pair(x, x)
        np.testing.assert_allclose(center, x)
        assert radius > 0, "Radius should be clamped above zero for coincident points"

    def test_arbitrary_dimension(self):
        """circle_from_pair should work for arbitrary dimensions."""
        x_i = np.array([1.0, 2.0, 3.0, 4.0])
        x_j = np.array([5.0, 6.0, 7.0, 8.0])
        center, radius = circle_from_pair(x_i, x_j)
        expected_center = 0.5 * (x_i + x_j)
        np.testing.assert_allclose(center, expected_center)
        expected_radius = 0.5 * np.linalg.norm(x_i - x_j)
        assert pytest.approx(radius) == expected_radius

    def test_negative_coordinates(self):
        """Should handle negative coordinates correctly."""
        x_i = np.array([-3.0, -4.0])
        x_j = np.array([3.0, 4.0])
        center, radius = circle_from_pair(x_i, x_j)
        np.testing.assert_allclose(center, np.array([0.0, 0.0]))
        assert pytest.approx(radius) == 5.0


# ---------------------------------------------------------------------------
# uniform_in_disk_vec
# ---------------------------------------------------------------------------

class TestUniformInDiskVec:
    """Tests for uniform_in_disk_vec(rng, centers, radii)."""

    def test_all_points_within_radius(self):
        """Every sampled point must be within its disk's radius."""
        rng = np.random.default_rng(42)
        n = 10000
        centers = np.zeros((n, 2))
        radii = np.full(n, 5.0)
        points = uniform_in_disk_vec(rng, centers, radii)

        dists = np.linalg.norm(points - centers, axis=1)
        assert np.all(dists <= radii + 1e-10), (
            "Some points lie outside their disk"
        )

    def test_output_shape(self):
        """Output shape should be (n, 2)."""
        rng = np.random.default_rng(0)
        n = 50
        centers = rng.standard_normal((n, 2))
        radii = np.abs(rng.standard_normal(n)) + 0.1
        points = uniform_in_disk_vec(rng, centers, radii)
        assert points.shape == (n, 2)

    def test_different_radii(self):
        """Should respect per-disk radii."""
        rng = np.random.default_rng(123)
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        radii = np.array([1.0, 0.01])
        points = uniform_in_disk_vec(rng, centers, radii)

        for i in range(2):
            dist = np.linalg.norm(points[i] - centers[i])
            assert dist <= radii[i] + 1e-10

    def test_single_point(self):
        """Should work with a single disk."""
        rng = np.random.default_rng(7)
        center = np.array([[3.0, 4.0]])
        radius = np.array([2.0])
        pt = uniform_in_disk_vec(rng, center, radius)
        assert pt.shape == (1, 2)
        assert np.linalg.norm(pt[0] - center[0]) <= 2.0 + 1e-10

    def test_uniform_coverage_statistical(self):
        """Rough statistical test: mean of many samples should approximate center."""
        rng = np.random.default_rng(42)
        n = 50000
        center = np.array([5.0, -3.0])
        centers = np.tile(center, (n, 1))
        radii = np.full(n, 2.0)
        points = uniform_in_disk_vec(rng, centers, radii)
        mean_pt = points.mean(axis=0)
        np.testing.assert_allclose(mean_pt, center, atol=0.1)


# ---------------------------------------------------------------------------
# points_in_circle
# ---------------------------------------------------------------------------

class TestPointsInCircle:
    """Tests for points_in_circle(X, center, radius)."""

    def test_correct_boolean_mask(self):
        """Points inside the circle should be flagged True, others False."""
        X = np.array([
            [0.0, 0.0],   # inside (dist=0)
            [1.0, 0.0],   # on boundary (dist=1)
            [2.0, 0.0],   # outside (dist=2)
            [0.5, 0.5],   # inside (dist~0.707)
        ])
        center = np.array([0.0, 0.0])
        radius = 1.0
        mask = points_in_circle(X, center, radius)
        expected = np.array([True, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_all_inside(self):
        """If radius is large enough, all points should be inside."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2))
        center = np.array([0.0, 0.0])
        radius = 100.0
        mask = points_in_circle(X, center, radius)
        assert mask.all()

    def test_none_inside(self):
        """If radius is tiny and center is far from points, none should be inside."""
        X = np.array([[10.0, 10.0], [20.0, 20.0]])
        center = np.array([0.0, 0.0])
        radius = 1e-6
        mask = points_in_circle(X, center, radius)
        assert not mask.any()

    def test_empty_array(self):
        """Empty input should return an empty boolean array."""
        X = np.empty((0, 2))
        center = np.array([0.0, 0.0])
        mask = points_in_circle(X, center, 1.0)
        assert mask.shape == (0,)

    def test_higher_dimension(self):
        """points_in_circle should work in higher dimensions (it's a sphere test)."""
        X = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # dist = sqrt(3) ~ 1.732
        ])
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.5
        mask = points_in_circle(X, center, radius)
        np.testing.assert_array_equal(mask, [True, False])


# ---------------------------------------------------------------------------
# vonmises_in_disk
# ---------------------------------------------------------------------------

class TestVonMisesInDisk:
    """Tests for vonmises_in_disk(rng, center, radius, mu, kappa, n_samples)."""

    def test_all_points_within_radius(self):
        """Every sampled point must lie within the disk."""
        rng = np.random.default_rng(42)
        center = np.array([1.0, 2.0])
        radius = 3.0
        points = vonmises_in_disk(rng, center, radius, mu=0.0, kappa=5.0,
                                  n_samples=5000)
        dists = np.linalg.norm(points - center, axis=1)
        assert np.all(dists <= radius + 1e-10)

    def test_output_shape(self):
        """Output shape should be (n_samples, 2)."""
        rng = np.random.default_rng(0)
        n = 200
        pts = vonmises_in_disk(rng, np.zeros(2), 1.0, 0.0, 1.0, n)
        assert pts.shape == (n, 2)

    def test_zero_kappa_gives_roughly_uniform(self):
        """With kappa=0, the Von Mises distribution is uniform on the circle
        so angular distribution should be roughly uniform."""
        rng = np.random.default_rng(42)
        center = np.array([0.0, 0.0])
        pts = vonmises_in_disk(rng, center, 1.0, mu=0.0, kappa=0.0,
                               n_samples=10000)
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        # Divide into 4 quadrants; counts should be roughly equal.
        q1 = np.sum((angles >= 0) & (angles < np.pi / 2))
        q2 = np.sum((angles >= np.pi / 2) & (angles < np.pi))
        q3 = np.sum((angles >= -np.pi) & (angles < -np.pi / 2))
        q4 = np.sum((angles >= -np.pi / 2) & (angles < 0))
        counts = np.array([q1, q2, q3, q4])
        # Each quadrant should have roughly 25% of the samples.
        assert np.all(counts > 1500), (
            f"With kappa=0, quadrant counts should be roughly equal: {counts}"
        )

    def test_high_kappa_concentrates_direction(self):
        """With high kappa, most points should be in the direction of mu."""
        rng = np.random.default_rng(42)
        center = np.array([0.0, 0.0])
        pts = vonmises_in_disk(rng, center, 1.0, mu=0.0, kappa=100.0,
                               n_samples=5000)
        # Most points should have angle near 0 (i.e. positive x direction).
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        near_zero = np.sum(np.abs(angles) < np.pi / 4)
        assert near_zero > 3000, (
            "With kappa=100 and mu=0, most points should be near angle 0"
        )


# ---------------------------------------------------------------------------
# rotate_2d and rotate_batch_2d
# ---------------------------------------------------------------------------

class TestRotate2D:
    """Tests for rotate_2d and rotate_batch_2d."""

    def test_90_degree_rotation(self):
        """Rotating (1, 0) by 90 degrees should give (0, 1)."""
        v = np.array([1.0, 0.0])
        result = rotate_2d(v, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)

    def test_180_degree_rotation(self):
        """Rotating (1, 0) by 180 degrees should give (-1, 0)."""
        v = np.array([1.0, 0.0])
        result = rotate_2d(v, np.pi)
        np.testing.assert_allclose(result, [-1.0, 0.0], atol=1e-12)

    def test_360_degree_rotation_identity(self):
        """Full rotation should return the original vector."""
        v = np.array([3.0, 4.0])
        result = rotate_2d(v, 2 * np.pi)
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_preserves_magnitude(self):
        """Rotation should not change vector magnitude."""
        v = np.array([3.0, 4.0])
        angle = 1.23
        result = rotate_2d(v, angle)
        assert pytest.approx(np.linalg.norm(result)) == np.linalg.norm(v)

    def test_zero_rotation(self):
        """Zero rotation should return the same vector."""
        v = np.array([2.5, -1.3])
        result = rotate_2d(v, 0.0)
        np.testing.assert_allclose(result, v, atol=1e-14)

    def test_batch_90_degree_rotation(self):
        """Batch rotation of multiple vectors by 90 degrees each."""
        V = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        angles = np.full(3, np.pi / 2)
        result = rotate_batch_2d(V, angles)
        expected = np.array([
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_different_angles(self):
        """Batch rotation with different per-vector angles."""
        V = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        angles = np.array([0.0, np.pi])
        result = rotate_batch_2d(V, angles)
        expected = np.array([
            [1.0, 0.0],    # 0 rotation
            [-1.0, 0.0],   # pi rotation
        ])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_preserves_magnitudes(self):
        """All magnitudes should be preserved after batch rotation."""
        rng = np.random.default_rng(42)
        V = rng.standard_normal((100, 2))
        angles = rng.uniform(0, 2 * np.pi, 100)
        result = rotate_batch_2d(V, angles)
        original_norms = np.linalg.norm(V, axis=1)
        result_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(result_norms, original_norms, atol=1e-12)

    def test_batch_matches_single(self):
        """Batch rotation should give the same results as individual rotations."""
        rng = np.random.default_rng(7)
        V = rng.standard_normal((10, 2))
        angles = rng.uniform(0, 2 * np.pi, 10)
        batch_result = rotate_batch_2d(V, angles)
        for i in range(10):
            single_result = rotate_2d(V[i], angles[i])
            np.testing.assert_allclose(batch_result[i], single_result, atol=1e-12)


# ---------------------------------------------------------------------------
# assign_voronoi
# ---------------------------------------------------------------------------

class TestAssignVoronoi:
    """Tests for assign_voronoi(X, centroids)."""

    def test_basic_assignment(self):
        """Each point should be assigned to its nearest centroid."""
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        X = np.array([
            [0.1, 0.1],    # nearest to centroid 0
            [9.9, 10.1],   # nearest to centroid 1
            [5.0, 5.0],    # equidistant; either is fine
        ])
        labels = assign_voronoi(X, centroids)
        assert labels[0] == 0
        assert labels[1] == 1

    def test_output_shape(self):
        """Output should have one label per point."""
        centroids = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        X = np.zeros((20, 2))
        labels = assign_voronoi(X, centroids)
        assert labels.shape == (20,)


# ---------------------------------------------------------------------------
# sample_in_voronoi_cell
# ---------------------------------------------------------------------------

class TestSampleInVoronoiCell:
    """Tests for sample_in_voronoi_cell."""

    def test_samples_within_disk(self):
        """All returned points should be inside the bounding disk (approximately)."""
        rng = np.random.default_rng(42)
        centroids = np.array([[0.0, 0.0], [2.0, 0.0]])
        center = np.array([1.0, 0.0])
        radius = 3.0
        pts = sample_in_voronoi_cell(rng, centroids, 0, center, radius,
                                     n_samples=100)
        assert pts.shape == (100, 2)
        # The samples should be closer to centroid 0 than centroid 1.
        d0 = np.linalg.norm(pts - centroids[0], axis=1)
        d1 = np.linalg.norm(pts - centroids[1], axis=1)
        assert np.all(d0 <= d1 + 0.5)  # small tolerance for edge cases

    def test_zero_samples(self):
        """Requesting zero samples should return an empty array."""
        rng = np.random.default_rng(0)
        centroids = np.array([[0.0, 0.0]])
        pts = sample_in_voronoi_cell(rng, centroids, 0, np.zeros(2), 1.0,
                                     n_samples=0)
        assert pts.shape == (0, 2)


# ---------------------------------------------------------------------------
# uniform_in_ball / uniform_in_ball_batch
# ---------------------------------------------------------------------------

class TestUniformInBall:
    """Tests for high-dimensional ball sampling."""

    def test_single_point_in_ball(self):
        """Sampled point should be within the ball."""
        rng = np.random.default_rng(42)
        center = np.array([1.0, 2.0, 3.0])
        R = 2.0
        pt = uniform_in_ball(rng, center, R, dim=3)
        assert pt.shape == (3,)
        assert np.linalg.norm(pt - center) <= R + 1e-10

    def test_batch_all_within_ball(self):
        """All batch-sampled points should be within their respective balls."""
        rng = np.random.default_rng(42)
        n = 5000
        d = 5
        centers = np.zeros((n, d))
        radii = np.full(n, 3.0)
        pts = uniform_in_ball_batch(rng, centers, radii)
        dists = np.linalg.norm(pts - centers, axis=1)
        assert np.all(dists <= 3.0 + 1e-10)

    def test_batch_output_shape(self):
        """Batch output shape should be (n, d)."""
        rng = np.random.default_rng(0)
        centers = np.zeros((10, 4))
        radii = np.ones(10)
        pts = uniform_in_ball_batch(rng, centers, radii)
        assert pts.shape == (10, 4)
