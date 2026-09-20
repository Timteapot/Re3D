from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class ImageQuality:
    image_id: int
    frame_id: int
    name: str
    keypoints: int
    triangulated_observations: int

    @property
    def triangulated_ratio(self) -> float:
        if self.keypoints <= 0:
            return 0.0
        return self.triangulated_observations / self.keypoints

    def to_dict(self) -> dict[str, int | float | str]:
        result = asdict(self)
        result["triangulated_ratio"] = self.triangulated_ratio
        return result


def reconstruction_image_quality(reconstruction: Any) -> list[ImageQuality]:
    result: list[ImageQuality] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        result.append(
            ImageQuality(
                image_id=int(image_id),
                frame_id=int(image.frame_id),
                name=str(image.name),
                keypoints=int(image.num_points2D()),
                triangulated_observations=int(image.num_points3D),
            )
        )
    return result


def select_deferred_images(
    images: Iterable[ImageQuality],
    *,
    relative_observation_floor: float,
    minimum_triangulated_ratio: float,
    max_images: int,
) -> tuple[list[ImageQuality], dict[str, float | int]]:
    values = list(images)
    if not values:
        return [], {
            "registered_images": 0,
            "median_triangulated_observations": 0.0,
            "observation_limit": 0.0,
        }
    if not 0 < relative_observation_floor < 1:
        raise ValueError("relative_observation_floor must be between 0 and 1")
    if not 0 < minimum_triangulated_ratio < 1:
        raise ValueError("minimum_triangulated_ratio must be between 0 and 1")
    if max_images < 1:
        raise ValueError("max_images must be >= 1")

    median_observations = float(median(item.triangulated_observations for item in values))
    observation_limit = median_observations * relative_observation_floor
    candidates = [
        item
        for item in values
        if item.triangulated_observations < observation_limit
        and item.triangulated_ratio < minimum_triangulated_ratio
    ]
    candidates.sort(key=lambda item: (item.triangulated_ratio, item.triangulated_observations))
    return candidates[:max_images], {
        "registered_images": len(values),
        "median_triangulated_observations": median_observations,
        "observation_limit": observation_limit,
    }
