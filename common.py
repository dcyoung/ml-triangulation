import numpy.typing as npt
import numpy as np
from pyproj import Transformer
from functools import reduce
from tqdm import tqdm
import operator

RADIUS_EARTH_KM = 6373.0
RADIUS_EARTH_M = RADIUS_EARTH_KM * 1000
TRANS_GPS_TO_XYZ = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


def enforce_safe_lat_long(lat_long_deg: npt.NDArray) -> npt.NDArray:
    """
    Normalize point to [-90, 90] latitude and [-180, 180] longitude.
    """
    lat_long_deg = 1.0 * lat_long_deg
    lat_long_deg[:, 0] = (lat_long_deg[:, 0] + 90) % 360 - 90
    mask = lat_long_deg[:, 0] > 90
    lat_long_deg[mask, 0] = 180 - lat_long_deg[mask, 0]
    lat_long_deg[mask, 1] += 180
    lat_long_deg[:, 1] = (lat_long_deg[:, 1] + 180) % 360 - 180
    return lat_long_deg


# Assuming normalized lat/long contains values in range -1:1... return de-normalized values as true lat/long
def denorm_lat_long(normalized_lat_long: npt.NDArray) -> npt.NDArray:
    lat_long = (normalized_lat_long + 1.0) / 2.0
    lat_long[:, 0] *= 180.0
    lat_long[:, 0] -= 90.0
    lat_long[:, 1] *= 360.0
    lat_long[:, 1] -= 180.0
    return enforce_safe_lat_long(lat_long)


def normalize_lat_long(lat_long_deg: npt.NDArray) -> npt.NDArray:
    """Normalizes lat/long to range -1:1"""
    lat_long = enforce_safe_lat_long(lat_long_deg=lat_long_deg)
    lat_long[:, 0] = 1 - 2 * (lat_long[:, 0] + 90.0) / 180.0
    lat_long[:, 1] = 1 - 2 * (lat_long[:, 1] + 180.0) / 360.0
    return lat_long


def haversine_np(
    lat_long_deg_1: npt.NDArray,
    lat_long_deg_2: npt.NDArray,
    radius: float = RADIUS_EARTH_KM,
) -> npt.NDArray:
    """
    Calculate the great circle distance between two points on a sphere
    ie: Shortest distance between two points on the surface of a sphere
    """
    lat_1, lon_1, lat_2, lon_2 = map(
        np.deg2rad,
        [
            lat_long_deg_1[:, 0],
            lat_long_deg_1[:, 1],
            lat_long_deg_2[:, 0],
            lat_long_deg_2[:, 1],
        ],
    )
    d = (
        np.sin((lat_2 - lat_1) / 2) ** 2
        + np.cos(lat_1) * np.cos(lat_2) * np.sin((lon_2 - lon_1) / 2) ** 2
    )
    arc_len = 2 * radius * np.arcsin(np.sqrt(d))
    return arc_len


def equirectangular_np(
    lat_long_deg_1: npt.NDArray, lat_long_deg_2: npt.NDArray, radius: float = 1.0
) -> npt.NDArray:
    """
    Simplified version of haversine for small distances where
    Pythagoras theorem can be used on an equirectangular projection
    """
    lat_1, lon_1, lat_2, lon_2 = map(
        np.deg2rad,
        [
            lat_long_deg_1[:, 0],
            lat_long_deg_1[:, 1],
            lat_long_deg_2[:, 0],
            lat_long_deg_2[:, 1],
        ],
    )
    x = (lon_2 - lon_1) * np.cos(0.5 * (lat_2 - lat_1))
    y = lat_2 - lat_1
    return radius * np.sqrt(x * x + y * y)


def geo_to_cartesian_m(lat_long_alt: npt.NDArray) -> npt.NDArray:
    """Converts lat/long/altitude to cartesian coordinates (in meters)

    Args:
        lat_long_alt (npt.NDArray): geo coordinates (n_samples, 3)

    Returns:
        npt.NDArray: cartesian coordinates in meters (n_samples, 3)
    """
    return np.stack(
        list(
            TRANS_GPS_TO_XYZ.transform(
                lat_long_alt[:, 1], lat_long_alt[:, 0], lat_long_alt[:, 2]
            )
        ),
        axis=1,
    )


def cartesian_to_geo(xyz_m: npt.NDArray) -> npt.NDArray:
    """Converts cartesian coordinates (m) to lat/long/altitude

    Args:
        xyz_m (npt.NDArray): cartesian coordinates in meters (n_samples, 3)

    Returns:
        npt.NDArray: geo coordinates (lat/long/alt) (n_samples, 3)
    """
    long, lat, alt = TRANS_GPS_TO_XYZ.transform(
        xyz_m[:, 0],
        xyz_m[:, 1],
        xyz_m[:, 2],
        direction="INVERSE",
    )
    return np.stack([lat, long, alt], axis=1)


def calculate_centroids(
    coordinates_xyz_m: npt.NDArray,
    project: bool = True,
) -> npt.NDArray:
    """
    Calculate the centroids for each sample. A centroid is the center of mass on \
        the surface of the earth between the various coordinates. For relatively small \
        distances (nearby points on a sphere), a projection of the cartesian \
        centroid is probably fine

    Args:
        coordinates_xyz_m (npt.NDArray): (n_coords_per_sample, n_samples, 3)
        project (bool): if True, project the coordinate to the surface of the earth... otherwise returns a purely cartesian centroid
    Returns:
        npt.NDArray: (n_samples, 3)
    """
    # Make sure inputs are as expected
    n_coords_per_sample, n_samples, n_dim = coordinates_xyz_m.shape
    assert n_dim == 3  # each coord should be 3 dimensional (xyz)

    # calculate the cartesian centroid (this will be INSIDE the sphere)
    cartesian_centroid_xyz_m = reduce(operator.add, tuple(coordinates_xyz_m)) / n_dim

    if not project:
        return cartesian_centroid_xyz_m

    # convert to lat/long - will have a negative altitude
    centroid_long, centroid_lat, _ = TRANS_GPS_TO_XYZ.transform(
        cartesian_centroid_xyz_m[:, 0],
        cartesian_centroid_xyz_m[:, 1],
        cartesian_centroid_xyz_m[:, 2],
        direction="INVERSE",
    )

    # ZERO out the altitude (ie: on the surface)
    lat_long_alt = np.stack(
        [centroid_lat, centroid_long, np.zeros_like(centroid_lat)], axis=1
    )

    # Project back out to the surface of the sphere by again calculating x,y,z from the same lat/long but with ZERO altitude (ie: on the surface)
    projected_centroid_xyz_m = geo_to_cartesian_m(lat_long_alt=lat_long_alt)

    return projected_centroid_xyz_m


def fspl_distance(rssi: npt.NDArray, frequency_mhz: npt.NDArray) -> npt.NDArray:
    """Calculates fspl distance using signal strengths

    Args:
        rssi (npt.NDArray): the signal strength
        frequency_mhz (npt.NDArray): the frequency of the signal

    Returns:
        npt.NDArray: the inferred distance (in km)
    """
    return 10 ** ((np.abs(rssi) - 32.45 - 20 * np.log10(frequency_mhz)) / 20)


def str_contains_sub(s: str, subs) -> bool:
    if isinstance(subs, str):
        subs = [subs]
    for sub in subs:
        if sub in s:
            return True
    return False


def haversine_cluster(
    points_lat_long_deg: npt.NDArray,
    centroids_lat_long_deg: npt.NDArray,
    trace: bool = False,
) -> npt.NDArray:
    """Cluster points to the closest centroid based on haversine dist

    Args:
        points_lat_long_deg (npt.NDArray): the data points to cluster, shape (n, 2)
        centroids_lat_long_deg (npt.NDArray): the cluster centroids, shape (k, 2)
        trace (bool, optional): If True, display progress bar. Defaults to True.

    Returns:
        (npt.NDArray): labels (cluster indices) for each data point
    """
    # Cluster the data points to the nearest "cluster" based on haversine dist
    n = points_lat_long_deg.shape[0]
    k = centroids_lat_long_deg.shape[0]
    # Assign centroids based on minimum haversine distance
    diff = np.zeros((n, k))
    for i in tqdm(range(k), disable=not trace):
        diff[:, i] = haversine_np(
            points_lat_long_deg, centroids_lat_long_deg[np.newaxis, i, :]
        )
    labels = diff.argmin(axis=1)  # n,
    return labels
