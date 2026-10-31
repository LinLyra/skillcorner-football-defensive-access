import pytest

from defensive_access.geometry import convex_hull_area, point_segment_distance


def test_point_segment_distance_and_projection():
    distance, projection = point_segment_distance((5, 2), (0, 0), (10, 0))
    assert distance == pytest.approx(2.0)
    assert projection == pytest.approx(0.5)


def test_convex_hull_area_ignores_interior_points():
    points = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    assert convex_hull_area(points) == pytest.approx(4.0)

