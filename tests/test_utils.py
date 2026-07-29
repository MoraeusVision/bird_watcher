import numpy as np

from utils import crop_image


def test_crop_image_returns_requested_region() -> None:
    frame = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)

    cropped = crop_image(frame, (1, 1, 4, 3))

    assert cropped.shape == (2, 3)
    np.testing.assert_array_equal(cropped, frame[1:3, 1:4])


def test_crop_image_clips_out_of_bounds_coordinates() -> None:
    frame = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)

    cropped = crop_image(frame, (-2, -1, 3, 2))

    assert cropped.shape == (2, 3)
    np.testing.assert_array_equal(cropped, frame[0:2, 0:3])