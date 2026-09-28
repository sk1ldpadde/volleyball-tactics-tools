import numpy as np

from sports.common.view import ViewTransformer, get_bottom_center_points


def test_bounding_box_ground_point_uses_bottom_center() -> None:
    boxes = np.asarray([[10.0, 20.0, 30.0, 80.0]], dtype=np.float32)

    np.testing.assert_array_equal(get_bottom_center_points(boxes), [(20.0, 80.0)])


def test_corners_center_and_inverse_transform() -> None:
    image_points = np.asarray(
        [(220.0, 110.0), (740.0, 140.0), (900.0, 650.0), (80.0, 620.0)],
        dtype=np.float32,
    )
    court_points = np.asarray(
        [(0.0, 0.0), (9.0, 0.0), (9.0, 18.0), (0.0, 18.0)],
        dtype=np.float32,
    )
    transformer = ViewTransformer(image_points, court_points)

    np.testing.assert_allclose(
        transformer.transform_points(image_points), court_points, atol=1e-4)
    image_center = transformer.inverse_transform_points(
        np.asarray([(4.5, 9.0)], dtype=np.float32))
    np.testing.assert_allclose(
        transformer.transform_points(image_center), [(4.5, 9.0)], atol=1e-4)
    np.testing.assert_allclose(
        transformer.inverse_transform_points(court_points), image_points, atol=1e-3)


def test_homography_extrapolates_outside_court() -> None:
    transformer = ViewTransformer(
        np.asarray([(0, 0), (90, 0), (90, 180), (0, 180)], dtype=np.float32),
        np.asarray([(0, 0), (9, 0), (9, 18), (0, 18)], dtype=np.float32),
    )

    result = transformer.transform_points(
        np.asarray([(-20, -50), (120, 230)], dtype=np.float32))
    np.testing.assert_allclose(result, [(-2, -5), (12, 23)], atol=1e-5)
