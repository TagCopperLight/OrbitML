"""
World coastlines from Natural Earth (public domain), at 1:110 million.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).parents[1] / "data" / "coastline"
URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"


def load(min_spacing: float = 1.0, directory: Path = DIRECTORY) -> list[np.ndarray]:
    """Returns one (N, 2) array of [latitude, longitude] in degrees per coastline."""
    path = directory / Path(URL).name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)

    features = json.loads(path.read_text())["features"]
    # GeoJSON stores [longitude, latitude]
    return [thin(np.array(feature["geometry"]["coordinates"])[:, ::-1], min_spacing) for feature in features]


def thin(line: np.ndarray, min_spacing: float) -> np.ndarray:
    """Keeps the first and last points, and each point at least `min_spacing` degrees (of arc) from the last one kept."""
    latitude, longitude = np.radians(line).T
    units = np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=1)
    threshold = np.cos(np.radians(min_spacing))

    keep = [0]
    for i in range(1, len(line)):
        if units[i] @ units[keep[-1]] <= threshold:
            keep.append(i)
    if keep[-1] != len(line) - 1:
        keep.append(len(line) - 1)
    return line[keep]
