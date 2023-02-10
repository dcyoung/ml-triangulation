import numpy.typing as npt
import numpy as np
from pyproj import Transformer
from functools import reduce
import operator

TRANS_GPS_TO_XYZ = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


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
