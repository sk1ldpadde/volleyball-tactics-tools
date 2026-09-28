from typing import Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt


def get_bottom_center_points(
        xyxy: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
    """Return the ground/contact approximation for ``[x1, y1, x2, y2]`` boxes."""
    boxes = np.asarray(xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Bounding boxes must be an Nx4 array in xyxy format.")
    return np.column_stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3])
    ).astype(np.float32)


class ViewTransformer:
    def __init__(
            self,
            source: npt.NDArray[np.float32],
            target: npt.NDArray[np.float32],
            method: int = 0,
            ransac_reproj_threshold: float = 3.0,
            minimum_inliers: int = 4,
    ) -> None:
        """
        Initialize the ViewTransformer with source and target points.

        Args:
            source (npt.NDArray[np.float32]): Source points for homography calculation.
            target (npt.NDArray[np.float32]): Target points for homography calculation.

        Raises:
            ValueError: If source and target do not have the same shape or if they are
                not 2D coordinates.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("Source and target must be arrays of 2D coordinates.")
        if source.shape[0] < 4:
            raise ValueError("At least four point correspondences are required.")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Source and target points must be finite.")

        source = source.astype(np.float64)
        target = target.astype(np.float64)
        if np.unique(source, axis=0).shape[0] != source.shape[0]:
            raise ValueError("Source points must be unique.")
        if np.unique(target, axis=0).shape[0] != target.shape[0]:
            raise ValueError("Target points must be unique.")
        if np.linalg.matrix_rank(
                np.column_stack((source, np.ones(source.shape[0])))) < 3:
            raise ValueError("Source points are degenerate (collinear).")
        if np.linalg.matrix_rank(
                np.column_stack((target, np.ones(target.shape[0])))) < 3:
            raise ValueError("Target points are degenerate (collinear).")
        if ransac_reproj_threshold <= 0:
            raise ValueError("RANSAC reprojection threshold must be positive.")

        self.m, mask = cv2.findHomography(
            source,
            target,
            method=method,
            ransacReprojThreshold=ransac_reproj_threshold,
        )
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
        if not np.isfinite(self.m).all():
            raise ValueError("Homography matrix contains non-finite values.")
        scale = self.m[2, 2]
        if abs(scale) > np.finfo(float).eps:
            self.m = self.m / scale
        determinant = float(np.linalg.det(self.m))
        condition = float(np.linalg.cond(self.m))
        if abs(determinant) < 1e-12 or not np.isfinite(condition) or condition > 1e12:
            raise ValueError("Homography matrix is numerically unstable.")

        self.inlier_mask: Optional[npt.NDArray[np.bool_]] = (
            None if mask is None else mask.reshape(-1).astype(bool)
        )
        if self.inlier_mask is not None and int(self.inlier_mask.sum()) < minimum_inliers:
            raise ValueError(
                f"Homography retained only {int(self.inlier_mask.sum())} inliers; "
                f"at least {minimum_inliers} are required."
            )

        try:
            self.inverse_m = np.linalg.inv(self.m)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Homography matrix is not invertible.") from exc

    def transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Transform the given points using the homography matrix.

        Args:
            points (npt.NDArray[np.float32]): Points to be transformed.

        Returns:
            npt.NDArray[np.float32]: Transformed points.

        Raises:
            ValueError: If points are not 2D coordinates.
        """
        if points.size == 0:
            return points

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a two-dimensional array of 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def inverse_transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Transform target-space points back into source coordinates."""
        if points.size == 0:
            return points
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a two-dimensional array of 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.inverse_m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Transform the given image using the homography matrix.

        Args:
            image (npt.NDArray[np.uint8]): Image to be transformed.
            resolution_wh (Tuple[int, int]): Width and height of the output image.

        Returns:
            npt.NDArray[np.uint8]: Transformed image.

        Raises:
            ValueError: If the image is not either grayscale or color.
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")
        return cv2.warpPerspective(image, self.m, resolution_wh)
