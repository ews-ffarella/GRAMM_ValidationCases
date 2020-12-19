import io
import logging
import math
import struct
from pathlib import Path

import descartes
import fiona as fio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.coords
import rasterio.plot
import rasterio.transform
import rasterio.windows
import shapely.geometry as sg
import vtk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba.tracing import noevent
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

log = logging.getLogger(__name__)


def readint(stream):
    return struct.unpack("i", stream.read(4))[0]


def readfloat(stream):
    return struct.unpack("f", stream.read(4))[0]


def readdouble(stream):
    return struct.unpack("d", stream.read(8))[0]


def find_exp_r(dx_ini, total_length, exp_ini, reverse=False):
    xl = dx_ini * (exp_ini ** np.arange(0, int(total_length / dx_ini) + 1, 1))
    nx_l = np.argmin(np.abs((xl.cumsum() - total_length))) - 1

    min_dx = -1
    exp = exp_ini
    new_xs = []
    cpt_iter = 0
    while min_dx < dx_ini:
        cpt_iter += 1
        exp = minimize(
            lambda exp_l: (
                total_length - dx_ini * ((1.0 - exp_l ** (nx_l)) / (1.0 - exp_l))
            )
            ** 2.0,
            (exp_ini,),
        ).x[0]
        xl = dx_ini * (exp ** np.arange(0, nx_l, 1))
        new_xs = xl * total_length / xl.sum()
        if reverse:
            new_xs = new_xs[::-1]
        min_dx = new_xs.min()
        print((cpt_iter, min_dx, nx_l))
        exp_ini = exp
        nx_l -= 1
    return exp, np.array(new_xs), new_xs[0] if reverse else new_xs[-1]


def generate_xys(resolution, mesh_zone, refinement_zone=None, refinement_grading=1.0):

    mesh_bbox = rio.coords.BoundingBox(*mesh_zone.bounds)
    mesh_bbox = rio.coords.BoundingBox(
        resolution * np.floor(mesh_bbox.left / resolution),
        resolution * np.floor(mesh_bbox.bottom / resolution),
        resolution * np.ceil(mesh_bbox.right / resolution),
        resolution * np.ceil(mesh_bbox.top / resolution),
    )
    ref_bbox = None
    if refinement_zone is not None:
        ref_bbox = rio.coords.BoundingBox(*refinement_zone.bounds)
        ref_bbox = rio.coords.BoundingBox(
            resolution * np.floor(ref_bbox.left / resolution),
            resolution * np.floor(ref_bbox.bottom / resolution),
            resolution * np.ceil(ref_bbox.right / resolution),
            resolution * np.ceil(ref_bbox.top / resolution),
        )
        assert sg.box(*mesh_bbox).contains(sg.box(*ref_bbox))

    nx = int((mesh_bbox.right - mesh_bbox.left) / resolution)
    ny = int((mesh_bbox.top - mesh_bbox.bottom) / resolution)

    xs = ys = None

    if ref_bbox is not None:
        nx_r = int((ref_bbox.right - ref_bbox.left) / resolution)
        ny_r = int((ref_bbox.top - ref_bbox.bottom) / resolution)
        r_l = ref_bbox.left - mesh_bbox.left
        r_b = ref_bbox.bottom - mesh_bbox.bottom
        r_r = mesh_bbox.right - ref_bbox.right
        r_t = mesh_bbox.top - ref_bbox.top

        exp_l, cells_l, last_l = find_exp_r(
            resolution, r_l, refinement_grading, reverse=True
        )
        exp_b, cells_b, last_b = find_exp_r(
            resolution, r_b, refinement_grading, reverse=True
        )
        exp_r, cells_r, last_r = find_exp_r(resolution, r_r, refinement_grading)
        exp_t, cells_t, last_t = find_exp_r(resolution, r_t, refinement_grading)

        dist_x = np.array(
            [0] + cells_l.tolist() + [resolution] * nx_r + cells_r.tolist()
        )
        dist_y = np.array(
            [0] + cells_b.tolist() + [resolution] * ny_r + cells_t.tolist()
        )

        xs = mesh_bbox.left + dist_x.cumsum()
        ys = mesh_bbox.bottom + dist_y.cumsum()
        assert np.allclose(xs[-1], mesh_bbox.right)
        assert np.allclose(xs[0], mesh_bbox.left)
        assert np.allclose(ys[-1], mesh_bbox.top)
        assert np.allclose(ys[0], mesh_bbox.bottom)
        mins = (cells_l.min(), cells_r.min(), cells_b.min(), cells_t.min())
        print(mins, max(*mins))
        lasts = (last_l, last_b, last_r, last_t)
        print(lasts, max(*lasts))
    else:
        xs = np.arange(mesh_bbox.left, mesh_bbox.right + 0.1 * resolution, resolution)
        ys = np.arange(mesh_bbox.bottom, mesh_bbox.top + 0.1 * resolution, resolution)
        dist_x = np.ones(xs.size) * resolution
        dist_x[0] = 0
        dist_y = np.ones(ys.size) * resolution
        dist_y[0] = 0

    assert np.allclose(dist_x.cumsum()[-1], mesh_bbox.right - mesh_bbox.left)
    assert np.allclose(dist_y.cumsum()[-1], mesh_bbox.top - mesh_bbox.bottom)

    pts_df = np.column_stack(_.flatten() for _ in np.meshgrid(xs, ys))
    pts_df = pd.DataFrame(pts_df, columns=["X", "Y"])
    nx = xs.size - 1
    ny = ys.size - 1
    return pts_df, nx, ny, xs, ys, dist_x, dist_y, mesh_bbox, ref_bbox


def data_window(src_data, src_transform, window_bounds):
    gtransform = rasterio.transform.guard_transform(src_transform)
    src_height, src_width = src_data.shape
    win = rio.windows.from_bounds(
        *window_bounds,
        transform=src_transform,
        height=None,
        width=None,
        precision=None,
    )
    window_floored = win.round_offsets(op="floor", pixel_precision=None)
    w = math.ceil(win.width + win.col_off - window_floored.col_off)
    h = math.ceil(win.height + win.row_off - window_floored.row_off)
    win = rasterio.windows.Window(window_floored.col_off, window_floored.row_off, w, h)
    row_off, col_off = win.row_off, win.col_off
    if row_off < 0 or col_off < 0 or row_off >= src_height or col_off >= src_width:
        win = rasterio.windows.from_bounds(
            *window_bounds,
            transform=src_transform,
            height=src_height,
            width=src_width,
            precision=None,
        )
        window_floored = win.round_offsets(op="floor", pixel_precision=None)
        w = math.ceil(win.width + win.col_off - window_floored.col_off)
        h = math.ceil(win.height + win.row_off - window_floored.row_off)
        win = rasterio.windows.Window(
            window_floored.col_off, window_floored.row_off, w, h
        )
        row_off, col_off = win.row_off, win.col_off
        if row_off < 0 or col_off < 0 or row_off >= src_height or col_off >= src_width:
            raise RuntimeError

    transform = rasterio.windows.transform(win, gtransform)
    slice_x, slice_y = win.toranges()
    data = src_data[slice(*slice_x), slice(*slice_y)]
    extent = rasterio.plot.plotting_extent(data, transform)
    return data, transform, extent, slice_x, slice_y


def create_bbox_interpolator(raster_filename, bbox, band=1, resolution=None):
    if not isinstance(raster_filename, (list, tuple)):
        with rio.open(raster_filename) as src:
            if resolution is None:
                resolution = max(*src.res)
            win = rio.windows.from_bounds(
                bbox.left - 1 * max(resolution, src.res[0]),
                bbox.bottom - 1 * max(resolution, src.res[1]),
                bbox.right + 1 * max(resolution, src.res[0]),
                bbox.top + 1 * max(resolution, src.res[1]),
                transform=src.transform,
                height=None,
                width=None,
                precision=None,
            )
            transform = rio.windows.transform(win, src.transform)
            data = src.read(band, masked=True, window=win)
            extent = rio.plot.plotting_extent(data, transform=transform)
            rx, ry = src.res
            assert data.shape == (int(win.height), int(win.width)), (
                data.shape,
                (int(win.height), int(win.width)),
            )
    else:
        src_data, src_transform = raster_filename
        rx, ry = src_transform[0], -src_transform[4]
        src_bounds = (
            bbox.left - 1 * max(resolution, rx),
            bbox.bottom - 1 * max(resolution, ry),
            bbox.right + 1 * max(resolution, rx),
            bbox.top + 1 * max(resolution, ry),
        )
        data, transform, extent, slice_x, slice_y = data_window(
            src_data, src_transform, src_bounds
        )

    height, width = data.shape
    rxs = (np.linspace(*extent[:2], width + 1) + 0.5 * transform[0])[:-1]
    rys = (np.linspace(*extent[2:], height + 1) + 0.5 * -transform[4])[:-1]

    interpolator = RegularGridInterpolator(
        (rxs, rys), data[::-1].T, bounds_error=False, fill_value=np.nan
    )
    return interpolator


LUT_KEYS = ["AGL", "Z0L", "EPSGL", "FWL", "ALAMBDAL", "ALAMBDAT"]


def make_arr(lut, data):
    vals = np.empty(max(lut.keys()) + 1)
    for k, v in lut.items():
        vals[k] = v
    return vals[data]


def generate_lancover_luts(clc_data):
    AGL = {}  # Albedo [%]
    EPSGL = {}  # Emissivity of the surface [%]
    FWL = {}  # Soil moisture [%]
    Z0L = {}  # Roughness length [m]
    ALAMBDAL = {}  # Heat conductivity [W/m" + SquareString]
    ALAMBDAT = {}  # Temperature conductivity [m/s]

    # 0 = Any default values (here forest 311)
    AGL[0] = 0.16
    EPSGL[0] = 0.95
    FWL[0] = 0.40
    Z0L[0] = 1.0
    ALAMBDAL[0] = 0.2
    ALAMBDAT[0] = 0.0000008
    # 111 = Continuous urban fabric
    AGL[111] = 0.25
    EPSGL[111] = 0.95
    FWL[111] = 0.03
    Z0L[111] = 1.0
    ALAMBDAL[111] = 1.0
    ALAMBDAT[111] = 0.000002
    # 112 = Discontinuous urban fabric
    AGL[112] = 0.25
    EPSGL[112] = 0.95
    FWL[112] = 0.03
    Z0L[112] = 0.5
    ALAMBDAL[112] = 1.0
    ALAMBDAT[112] = 0.0000013
    # 121 = Industrial or commercial units
    AGL[121] = 0.25
    EPSGL[121] = 0.95
    FWL[121] = 0.03
    Z0L[121] = 0.5
    ALAMBDAL[121] = 1.0
    ALAMBDAT[121] = 0.0000013
    # 122 = Road and rail networks and associated land
    AGL[122] = 0.25
    EPSGL[122] = 0.95
    FWL[122] = 0.03
    Z0L[122] = 0.3
    ALAMBDAL[122] = 1.0
    ALAMBDAT[122] = 0.0000013
    # 123 = Port areas
    AGL[123] = 0.25
    EPSGL[123] = 0.95
    FWL[123] = 0.03
    Z0L[123] = 1.0
    ALAMBDAL[123] = 1.0
    ALAMBDAT[123] = 0.0000013
    # 124 = Airports
    AGL[124] = 0.25
    EPSGL[124] = 0.95
    FWL[124] = 0.03
    Z0L[124] = 0.2
    ALAMBDAL[124] = 1.0
    ALAMBDAT[124] = 0.0000013
    # 131 = Mineral extraction sites
    AGL[131] = 0.25
    EPSGL[131] = 0.95
    FWL[131] = 0.03
    Z0L[131] = 0.2
    ALAMBDAL[131] = 1.0
    ALAMBDAT[131] = 0.0000013
    # 132 = Dump sites
    AGL[132] = 0.25
    EPSGL[132] = 0.95
    FWL[132] = 0.03
    Z0L[132] = 0.2
    ALAMBDAL[132] = 1.0
    ALAMBDAT[132] = 0.0000013
    # 133 = Construction sites
    AGL[133] = 0.25
    EPSGL[133] = 0.95
    FWL[133] = 0.03
    Z0L[133] = 0.2
    ALAMBDAL[133] = 1.0
    ALAMBDAT[133] = 0.0000013
    # 141 = Green urban areas
    AGL[141] = 0.19
    EPSGL[141] = 0.92
    FWL[141] = 0.10
    Z0L[141] = 0.3
    ALAMBDAL[141] = 0.2
    ALAMBDAT[141] = 0.0000007
    # 142 = Sport and leisure facilities
    AGL[142] = 0.19
    EPSGL[142] = 0.92
    FWL[142] = 0.10
    Z0L[142] = 0.3
    ALAMBDAL[142] = 0.2
    ALAMBDAT[142] = 0.0000007
    # 211 = Non-irrigated arable land
    AGL[211] = 0.19
    EPSGL[211] = 0.92
    FWL[211] = 0.10
    Z0L[211] = 0.1
    ALAMBDAL[211] = 0.2
    ALAMBDAT[211] = 0.0000007
    # 212 = Permanently-irrigated arable land
    AGL[212] = 0.19
    EPSGL[212] = 0.92
    FWL[212] = 0.50
    Z0L[212] = 0.1
    ALAMBDAL[212] = 1.0
    ALAMBDAT[212] = 0.0000007
    # 213 = Rice fields
    AGL[213] = 0.19
    EPSGL[213] = 0.92
    FWL[213] = 0.50
    Z0L[213] = 0.1
    ALAMBDAL[213] = 2.0
    ALAMBDAT[213] = 0.0000007
    # 221 = Vineyards
    AGL[221] = 0.19
    EPSGL[221] = 0.92
    FWL[221] = 0.10
    Z0L[221] = 0.15
    ALAMBDAL[221] = 0.2
    ALAMBDAT[221] = 0.0000007
    # 222 = Fruit trees and berry plantations
    AGL[222] = 0.19
    EPSGL[222] = 0.92
    FWL[222] = 0.10
    Z0L[222] = 0.25
    ALAMBDAL[222] = 0.2
    ALAMBDAT[222] = 0.0000007
    # 223 = Olive groves
    AGL[223] = 0.19
    EPSGL[223] = 0.92
    FWL[223] = 0.05
    Z0L[223] = 0.30
    ALAMBDAL[223] = 0.2
    ALAMBDAT[223] = 0.0000007
    # 231 = Pastures
    AGL[231] = 0.19
    EPSGL[231] = 0.92
    FWL[231] = 0.10
    Z0L[231] = 0.10
    ALAMBDAL[231] = 0.2
    ALAMBDAT[231] = 0.0000007
    # 241 = Annual crops associated with permanent crops
    AGL[241] = 0.19
    EPSGL[241] = 0.92
    FWL[241] = 0.10
    Z0L[241] = 0.10
    ALAMBDAL[241] = 0.2
    ALAMBDAT[241] = 0.0000007
    # 242 = Complex cultivation patterns
    AGL[242] = 0.19
    EPSGL[242] = 0.92
    FWL[242] = 0.10
    Z0L[242] = 0.20
    ALAMBDAL[242] = 0.2
    ALAMBDAT[242] = 0.0000007
    # 243 = Land principally occupied by agriculture, with significant areas of natural vegetation
    AGL[243] = 0.19
    EPSGL[243] = 0.92
    FWL[243] = 0.10
    Z0L[243] = 0.20
    ALAMBDAL[243] = 0.2
    ALAMBDAT[243] = 0.0000007
    # 244 = Agro-forestry areas
    AGL[244] = 0.17
    EPSGL[244] = 0.95
    FWL[244] = 0.40
    Z0L[244] = 1.0
    ALAMBDAL[244] = 0.2
    ALAMBDAT[244] = 0.0000008
    # 311 = Broad-leaved forest
    AGL[311] = 0.16
    EPSGL[311] = 0.95
    FWL[311] = 0.40
    Z0L[311] = 1.0
    ALAMBDAL[311] = 0.2
    ALAMBDAT[311] = 0.0000008
    # 312 = Coniferous forest
    AGL[312] = 0.12
    EPSGL[312] = 0.95
    FWL[312] = 0.40
    Z0L[312] = 1.0
    ALAMBDAL[312] = 0.2
    ALAMBDAT[312] = 0.0000008
    # 313 = Mixed forest
    AGL[313] = 0.14
    EPSGL[313] = 0.95
    FWL[313] = 0.40
    Z0L[313] = 1.0
    ALAMBDAL[313] = 0.2
    ALAMBDAT[313] = 0.0000008
    # 321 = Natural grasslands
    AGL[321] = 0.15
    EPSGL[321] = 0.92
    FWL[321] = 0.10
    Z0L[321] = 0.02
    ALAMBDAL[321] = 0.2
    ALAMBDAT[321] = 0.000001
    # 322 = Moors and heathland
    AGL[322] = 0.15
    EPSGL[322] = 0.92
    FWL[322] = 0.10
    Z0L[322] = 0.02
    ALAMBDAL[322] = 2.0
    ALAMBDAT[322] = 0.000001
    # 323 = Sclerophyllous vegeatation
    AGL[323] = 0.15
    EPSGL[323] = 0.92
    FWL[323] = 0.02
    Z0L[323] = 0.05
    ALAMBDAL[323] = 0.2
    ALAMBDAT[323] = 0.000001
    # 324 = Transitional woodland-shrub
    AGL[324] = 0.15
    EPSGL[324] = 0.92
    FWL[324] = 0.10
    Z0L[324] = 0.02
    ALAMBDAL[324] = 0.2
    ALAMBDAT[324] = 0.000001
    # 331 = Beaches, dunes, sands
    AGL[331] = 0.25
    EPSGL[331] = 0.95
    FWL[331] = 0.60
    Z0L[331] = 0.05
    ALAMBDAL[331] = 0.3
    ALAMBDAT[331] = 0.000001
    # 332 = Bare rocks
    AGL[332] = 0.15
    EPSGL[332] = 0.92
    FWL[332] = 0.01
    Z0L[332] = 0.10
    ALAMBDAL[332] = 1.0
    ALAMBDAT[332] = 0.000001
    # 333 = Sparsely vegetated areas
    AGL[333] = 0.15
    EPSGL[333] = 0.92
    FWL[333] = 0.01
    Z0L[333] = 0.01
    ALAMBDAL[333] = 0.2
    ALAMBDAT[333] = 0.000001
    # 334 = Burnt areas
    AGL[334] = 0.15
    EPSGL[334] = 0.92
    FWL[334] = 0.05
    Z0L[334] = 0.10
    ALAMBDAL[334] = 0.1
    ALAMBDAT[334] = 0.000001
    # 335 = Glaciers and perpetual snow
    AGL[335] = 0.60
    EPSGL[335] = 0.95
    FWL[335] = 0.10
    Z0L[335] = 0.01
    ALAMBDAL[335] = 1.0
    ALAMBDAT[335] = 0.0000005
    # 411 = Inland marshes
    AGL[411] = 0.14
    EPSGL[411] = 0.95
    FWL[411] = 0.70
    Z0L[411] = 0.01
    ALAMBDAL[411] = 20.0
    ALAMBDAT[411] = 0.000001
    # 412 = Peat bogs
    AGL[412] = 0.14
    EPSGL[412] = 0.95
    FWL[412] = 0.70
    Z0L[412] = 0.01
    ALAMBDAL[412] = 20.0
    ALAMBDAT[412] = 0.000001
    # 421 = Salt marshes
    AGL[421] = 0.50
    EPSGL[421] = 0.95
    FWL[421] = 0.70
    Z0L[421] = 0.01
    ALAMBDAL[421] = 20.0
    ALAMBDAT[421] = 0.000001
    # 422 = Salines
    AGL[422] = 0.50
    EPSGL[422] = 0.95
    FWL[422] = 0.70
    Z0L[422] = 0.01
    ALAMBDAL[422] = 20.0
    ALAMBDAT[422] = 0.000001
    # 423 = Intertidal flats
    AGL[423] = 0.14
    EPSGL[423] = 0.95
    FWL[423] = 0.70
    Z0L[423] = 0.01
    ALAMBDAL[423] = 20.0
    ALAMBDAT[423] = 0.000001
    # 511 = Water courses
    AGL[511] = 0.08
    EPSGL[511] = 0.98
    FWL[511] = 1.00
    Z0L[511] = 0.0001
    ALAMBDAL[511] = 100.0
    ALAMBDAT[511] = 0.000001
    # 512 = Water bodies
    AGL[512] = 0.08
    EPSGL[512] = 0.98
    FWL[512] = 1.00
    Z0L[512] = 0.0001
    ALAMBDAL[512] = 100.0
    ALAMBDAT[512] = 0.000001
    # 521 = Coastal lagoons
    AGL[521] = 0.081
    EPSGL[521] = 0.98
    FWL[521] = 1.00
    Z0L[521] = 0.0001
    ALAMBDAL[521] = 100.0
    ALAMBDAT[521] = 0.000001
    # 522 = Estuaries
    AGL[522] = 0.081
    EPSGL[522] = 0.98
    FWL[522] = 1.00
    Z0L[522] = 0.0001
    ALAMBDAL[522] = 100.0
    ALAMBDAT[522] = 0.000001
    # 523 = Sea and Ocean
    AGL[523] = 0.081
    EPSGL[523] = 0.98
    FWL[523] = 1.00
    Z0L[523] = 0.0001
    ALAMBDAL[523] = 100.0
    ALAMBDAT[523] = 0.000001

    unique_classes = np.unique(clc_data).tolist()
    assert len(set(unique_classes).difference(set(Z0L.keys()))) == 0

    landcover_luts = {
        "AGL": AGL,
        "Z0L": Z0L,
        "EPSGL": EPSGL,
        "FWL": FWL,
        "ALAMBDAL": ALAMBDAL,
        "ALAMBDAT": ALAMBDAT,
    }

    landcover_data = {"CLC": clc_data}
    for array_name in LUT_KEYS:
        landcover_data[array_name] = make_arr(landcover_luts[array_name], clc_data)

    return landcover_luts, landcover_data


def interpolate_landcover(
    raster_fname, xs, ys, landcover_luts, y_indices=None, resolution=None
):
    assert pd.Float64Index(xs)._is_strictly_monotonic_increasing
    assert pd.Float64Index(ys)._is_strictly_monotonic_increasing

    if y_indices is None:
        y_indices = list(range(len(ys)))

    nx = xs.size - 1
    ny = ys.size - 1
    assert ny > 0
    assert len(y_indices) > 0
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]

    with rio.open(raster_fname) as src:
        if resolution is None:
            resolution = max(*src.res)
        win = rio.windows.from_bounds(
            xmin - 2 * max(resolution, src.res[0]),
            ymin - 2 * max(resolution, src.res[1]),
            xmax + 2 * max(resolution, src.res[0]),
            ymax + 2 * max(resolution, src.res[1]),
            transform=src.transform,
            height=None,
            width=None,
            precision=None,
        )
        transform = rio.windows.transform(win, src.transform)
        data = src.read(1, masked=True, window=win).filled(0)
        rx, ry = src.res

    height, width = data.shape

    from functools import lru_cache

    @lru_cache(maxsize=(height + 1) * (width + 1))
    def make_geom(*args):
        return sg.box(*args)

    landcover_data = {"CLC": data}
    for array_name in LUT_KEYS:
        landcover_data[array_name] = make_arr(landcover_luts[array_name], data)

    # CLC, AGL, EPSGL, FWL, ALAMBDAL, ALAMBDAT
    ny = len(y_indices)
    land_cover_res = np.empty((nx, ny, len(LUT_KEYS) + 1))
    land_cover_dims = {i + 1: k for i, k in enumerate(LUT_KEYS)}

    y_idx = -1
    n_iters = len(y_indices)
    log.info(f"Starting processing {n_iters}")
    for y_i, y in enumerate(ys[:-1]):
        if y_i not in y_indices:
            continue
        y_idx += 1
        log.info(f"    {y_idx+1} / {n_iters}")
        for x_i, x in enumerate(xs[:-1]):
            bbox = rio.coords.BoundingBox(x, y, xs[x_i + 1], ys[y_i + 1])
            ox, oy = max(resolution, src.res[0]), max(resolution, src.res[1])
            pad_x = pad_y = 1

            b_bounds = (
                bbox.left - pad_x * ox,
                bbox.bottom - pad_y * oy,
                bbox.right + pad_x * ox,
                bbox.top + pad_y * oy,
            )

            b_arr, b_transform, b_extent, slice_x, slice_y = data_window(
                data, transform, b_bounds
            )
            assert b_arr.size > 0, bbox
            clc_vals = np.unique(b_arr)
            if clc_vals.size == 1:
                clc_val = clc_vals[0]
                land_cover_res[x_i, y_idx, 0] = clc_val
                for dim_i, array_name in land_cover_dims.items():
                    land_cover_res[x_i, y_idx, dim_i] = landcover_luts[array_name][
                        clc_val
                    ]
                continue

            b_h, b_w = b_arr.shape
            b_xs = np.linspace(*b_extent[:2], b_w + 1)
            b_ys = np.linspace(*b_extent[2:], b_h + 1)
            gs = np.column_stack(
                (
                    *[_.flatten() for _ in np.meshgrid(b_xs[:-1], b_ys[:-1])],
                    *[_.flatten() for _ in np.meshgrid(b_xs[1:], b_ys[1:])],
                )
            )
            gs = gpd.GeoSeries([make_geom(*tuple(_)) for _ in gs])
            areas = gs.intersection(sg.box(*bbox)).area
            fracs = (areas / areas.sum()).values
            main_clc = (
                pd.Series(fracs, index=b_arr.flatten()[::-1])
                .groupby(axis=0, level=0)
                .sum()
                .idxmax()
            )
            land_cover_res[x_i, y_idx, 0] = main_clc

            if 0:
                from scipy.interpolate import RegularGridInterpolator

                b_xs_o = b_xs[:-1] + src.res[0] * 0.5
                b_ys_o = b_ys[:-1] + src.res[1] * 0.5
                inter = RegularGridInterpolator(
                    (b_xs_o, b_ys_o),
                    b_arr[::-1].T,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                gs_i = np.column_stack(
                    [_.flatten() for _ in np.meshgrid(b_xs_o, b_ys_o)]
                )
                v = inter(gs_i)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                cm = plt.cm.Spectral_r
                norm = plt.Normalize(b_arr.min(), b_arr.max())
                ax1.scatter(gs_i[:, 0], gs_i[:, 1], color=cm(norm(v)))
                ax2.scatter(
                    gs_i[:, 0], gs_i[:, 1], color=cm(norm(b_arr[::-1].flatten()))
                )

            for dim_i, array_name in land_cover_dims.items():
                vals = landcover_data[array_name][slice(*slice_x), slice(*slice_y)][
                    ::-1
                ].flatten()
                land_cover_res[x_i, y_idx, dim_i] = np.sum(vals * fracs)
    res = []
    for yi, y_idx in enumerate(y_indices):
        res.append((y_idx, land_cover_res[:, yi, :]))
    return tuple(res)


class GrammMesher:
    def __init__(self):
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.xmin = 0.0
        self.ymin = 0.0
        self.xmax = 0.0
        self.ymax = 0.0
        self.dz0 = 0.0
        self.ddz = 0.0
        self.n_constant_cells = 0.0
        self.X = []
        self.Y = []
        self.Z = []
        self.AH = []
        self.VOL = []
        self.AREAX = []
        self.AREAY = []
        self.AREAZX = []
        self.AREAZY = []
        self.AREAZ = []
        self.ZSP = []
        self.DDX = []
        self.DDY = []
        self.ZAX = []
        self.ZAY = []
        self.AHE = []
        self.pts_df = None
        self.ccs_df = None

    def generate_geb(self):
        return f"""
{str(self.nx):20s}!Number of cells in x-direction
{str(self.ny):20s}!Number of cells in y-direction
{str(self.nz):20s}!Mumber of cells in z-direction
{str(self.xmin):20s}!West border of GRAMM model domain [m]
{str(self.xmax):20s}!East border of GRAMM model domain [m]
{str(self.ymin):20s}!South border of GRAMM model domain [m]
{str(self.ymax):20s}!North border of GRAMM model domain [m]
        """.strip()

    def write_landuse(self, filename):
        assert self.ccs_df.index.size == self.nx * self.ny
        RHOB = self.ccs_df["RHOB"].values
        ALAMBDA = self.ccs_df["ALAMBDA"].values
        Z0 = self.ccs_df["Z0"].values
        FW = self.ccs_df["FW"].values
        EPSG = self.ccs_df["EPSG"].values
        ALBEDO = self.ccs_df["ALBEDO"].values
        with open(str(filename), "w") as f:
            f.write(" ".join(RHOB.astype(np.int).astype(str).flatten(order="F")) + "\n")
            f.write(
                " ".join(ALAMBDA.round(3).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(Z0.round(4).astype(str).flatten(order="F")).replace(".0 ", " ")
                + "\n"
            )
            f.write(
                " ".join(FW.round(4).astype(str).flatten(order="F")).replace(".0 ", " ")
                + "\n"
            )
            f.write(
                " ".join(EPSG.round(4).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(ALBEDO.round(3).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )

        return filename

    @classmethod
    def create_mesh(
        cls,
        nx,
        ny,
        nz,
        pts_df,
        ccs_df,
        xs,
        ys,
        dist_x,
        dist_y,
        mesh_bbox,
        dz0,
        ddz,
        n_constant_cells=1,
        ah_min=None,
    ):

        xmin = mesh_bbox.left
        ymin = mesh_bbox.bottom
        xmax = mesh_bbox.right
        ymax = mesh_bbox.top

        assert pts_df.index.size == (nx + 1) * (ny + 1)
        assert ccs_df.index.size == nx * ny

        # Coordinates of points
        X = xs - xmin
        Y = ys - ymin

        AH = np.transpose(ccs_df["Z"].values.reshape(ny, nx, order="C"), [1, 0])
        # CLC = np.transpose(ccs_df['CLC_AreaWeighted'].values.reshape(ny, nx, order = 'C'), [1, 0])
        SurfaceEdge = np.transpose(
            pts_df["Z"].values.reshape(ny + 1, nx + 1, order="C"), [1, 0]
        )
        dh = SurfaceEdge.max() - SurfaceEdge.min()
        log.info(3.0 * dh)

        n_constant_cells = max(1, n_constant_cells)
        dz = dz0 * np.ones(n_constant_cells)
        dz = np.append(dz, dz0 * ddz ** np.arange(1, nz - n_constant_cells + 1))
        dz = np.append(0, np.cumsum(dz))

        # Points Locations
        ah_min_self = AH.min()
        log.info(f"AH_MIN: {ah_min_self}")
        if ah_min is None:
            ah_min = ah_min_self
        else:
            log.info(f"Using AH_MIN: {ah_min} (old {ah_min_self})")

        Z = ah_min + dz
        zmax = Z.max()
        # DDX, DDY = horizontal grid size in x and y directions
        DDX = np.copy(dist_x[1:])
        DDY = np.copy(dist_y[1:])
        # ZAX, ZAY = distance between cells in x and y directions
        ZAX = np.copy(dist_x[1:])
        ZAY = np.copy(dist_y[1:])
        ZAX[-1] = 0
        ZAY[-1] = 0

        VertStretch = (zmax - AH) / dz.max()
        ZSP = (
            AH[:, :, np.newaxis]
            + dz[np.newaxis, np.newaxis, :] * VertStretch[:, :, np.newaxis]
        )
        ZSP = (ZSP[:, :, :-1] + ZSP[:, :, 1:]) / 2.0

        VertStretch = (zmax - SurfaceEdge) / dz.max()
        AHE = (
            SurfaceEdge[:, :, np.newaxis]
            + dz[np.newaxis, np.newaxis, :] * VertStretch[:, :, np.newaxis]
        )

        # AREAS of the surfaces along x- and y- directions
        AREAX = (
            ((AHE[:, :-1, 1:] - AHE[:, :-1, :-1]) + (AHE[:, 1:, 1:] - AHE[:, 1:, :-1]))
            * 0.5
            * DDY[np.newaxis, :, np.newaxis]
        )
        AREAY = (
            ((AHE[:-1, :, 1:] - AHE[:-1, :, :-1]) + (AHE[1:, :, 1:] - AHE[1:, :, :-1]))
            * 0.5
            * DDX[:, np.newaxis, np.newaxis]
        )
        # AREAS of the ground projected along x- and y- directions
        AREAZX = (
            ((AHE[:-1, 1:, :] - AHE[1:, 1:, :]) + (AHE[:-1, :-1, :] - AHE[1:, :-1, :]))
            * 0.5
            * DDY[np.newaxis, :, np.newaxis]
        )
        AREAZY = (
            ((AHE[1:, :-1, :] - AHE[1:, 1:, :]) + (AHE[:-1, :-1, :] - AHE[:-1, 1:, :]))
            * 0.5
            * DDX[:, np.newaxis, np.newaxis]
        )
        # Bottom area
        AREAZ = (
            DDX[:, np.newaxis, np.newaxis] ** 2 * DDY[np.newaxis, :, np.newaxis] ** 2
            + AREAZX ** 2
            + AREAZY ** 2
        )
        AREAZ = AREAZ ** 0.5
        # VOL: volume of cells
        VOL = (
            2 * AHE[:-1, :-1, 1:]
            + AHE[1:, :-1, 1:]
            + 2 * AHE[1:, 1:, 1:]
            + AHE[:-1, 1:, 1:]
        ) - (
            2 * AHE[:-1, :-1, :-1]
            + AHE[1:, :-1, :-1]
            + 2 * AHE[1:, 1:, :-1]
            + AHE[:-1, 1:, :-1]
        )
        VOL = (
            VOL / 6.0 * DDX[:, np.newaxis, np.newaxis] * DDY[np.newaxis, :, np.newaxis]
        )

        meshDef = cls()
        meshDef.pts_df = pts_df
        meshDef.ccs_df = ccs_df
        meshDef.nx = nx
        meshDef.ny = ny
        meshDef.nz = nz
        meshDef.xmin = xmin
        meshDef.ymin = ymin
        meshDef.xmax = xmax
        meshDef.ymax = ymax
        meshDef.dz0 = dz0
        meshDef.ddz = ddz
        meshDef.n_constant_cells = n_constant_cells
        meshDef.X = X
        meshDef.Y = Y
        meshDef.Z = Z
        meshDef.AH = AH
        meshDef.VOL = VOL
        meshDef.AREAX = AREAX
        meshDef.AREAY = AREAY
        meshDef.AREAZ = AREAZ
        meshDef.AREAZX = AREAZX
        meshDef.AREAZY = AREAZY
        meshDef.ZSP = ZSP
        meshDef.DDX = DDX
        meshDef.DDY = DDY
        meshDef.ZAX = ZAX
        meshDef.ZAY = ZAY
        meshDef.AHE = AHE
        return meshDef

    @classmethod
    def read_mesh(cls, filename):
        self = cls()
        with open(str(filename), "rb") as fin:
            _ = fin.read(6)  # header
            self.nx = nx = readint(fin)
            self.ny = ny = readint(fin)
            self.nz = nz = readint(fin)
            self.AH = np.fromfile(fin, dtype="f", count=nx * ny, sep="")
            self.ZSP = np.fromfile(fin, dtype="f", count=nx * ny * nz, sep="")
            self.X = np.fromfile(fin, dtype="f", count=nx + 1, sep="")
            self.Y = np.fromfile(fin, dtype="f", count=ny + 1, sep="")
            self.Z = np.fromfile(fin, dtype="f", count=nz + 1, sep="")
            self.VOL = np.fromfile(fin, dtype="f", count=nx * ny * nz, sep="")
            self.AREAX = np.fromfile(fin, dtype="f", count=nz * ny * (nx + 1), sep="")
            self.AREAY = np.fromfile(fin, dtype="f", count=nz * nx * (ny + 1), sep="")
            self.AREAZX = np.fromfile(fin, dtype="f", count=ny * nx * (nz + 1), sep="")
            self.AREAZY = np.fromfile(fin, dtype="f", count=ny * nx * (nz + 1), sep="")
            self.AREAZ = np.fromfile(fin, dtype="f", count=ny * nx * (nz + 1), sep="")
            self.DDX = np.fromfile(fin, dtype="f", count=nx, sep="")
            self.DDY = np.fromfile(fin, dtype="f", count=ny, sep="")
            self.ZAX = np.fromfile(fin, dtype="f", count=nx, sep="")
            self.ZAY = np.fromfile(fin, dtype="f", count=ny, sep="")
            self.xmin = xmin = readint(fin)
            self.ymin = ymin = readint(fin)
            self.angle = angle = readdouble(fin)
            self.AHE = np.fromfile(
                fin, dtype="f", count=(nx + 1) * (ny + 1) * (nz + 1), sep=""
            )
            rest = np.fromfile(fin, dtype="f", sep="")
        assert rest.size == 0
        self.xmax = self.xmin + self.DDX.sum()
        self.ymax = self.ymin + self.DDY.sum()
        return self

    def write_ggeom_binary(self, filename, angle=0):
        with open(str(filename), "wb") as writebin:
            writebin.write(struct.pack("=B", 45))
            writebin.write(struct.pack("=B", 57))
            writebin.write(struct.pack("=B", 57))
            writebin.write(struct.pack("=B", 32))
            writebin.write(struct.pack("=B", 12))
            writebin.write(struct.pack("=B", 10))
            writebin.write(struct.pack("i", self.nx))
            writebin.write(struct.pack("i", self.ny))
            writebin.write(struct.pack("i", self.nz))
            writebin.write(struct.pack("=%sf" % self.AH.size, *self.AH.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.ZSP.size, *self.ZSP.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.X.size, *self.X.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.Y.size, *self.Y.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.Z.size, *self.Z.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.VOL.size, *self.VOL.flatten("F")))
            writebin.write(
                struct.pack("=%sf" % self.AREAX.size, *self.AREAX.flatten("F"))
            )
            writebin.write(
                struct.pack("=%sf" % self.AREAY.size, *self.AREAY.flatten("F"))
            )
            writebin.write(
                struct.pack("=%sf" % self.AREAZX.size, *self.AREAZX.flatten("F"))
            )
            writebin.write(
                struct.pack("=%sf" % self.AREAZY.size, *self.AREAZY.flatten("F"))
            )
            writebin.write(
                struct.pack("=%sf" % self.AREAZ.size, *self.AREAZ.flatten("F"))
            )
            writebin.write(struct.pack("=%sf" % self.DDX.size, *self.DDX.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.DDY.size, *self.DDY.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.ZAX.size, *self.ZAX.flatten("F")))
            writebin.write(struct.pack("=%sf" % self.ZAY.size, *self.ZAY.flatten("F")))
            writebin.write(struct.pack("i", int(self.xmin)))
            writebin.write(struct.pack("i", int(self.ymin)))
            writebin.write(struct.pack("d", float(angle)))
            writebin.write(struct.pack("=%sf" % self.AHE.size, *self.AHE.flatten("F")))
        return filename

    def write_ggeom_ascii(self, filename, angle=0):
        with open(str(filename), "w") as f:
            towrite = " ".join(map(str, [self.nx, self.ny, self.nz])) + " "
            towrite += " ".join(self.X.astype(np.int).astype(str)) + " "
            towrite += " ".join(self.Y.astype(np.int).astype(str)) + " "
            towrite += " ".join(self.Z.round(2).astype(str)) + " "
            f.write(towrite + "\n")
            f.write(
                " ".join(self.AH.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.VOL.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.AREAX.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.AREAY.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.AREAZX.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.AREAZY.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.AREAZ.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.ZSP.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.DDX.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.DDY.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.ZAX.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            f.write(
                " ".join(self.ZAY.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
                + "\n"
            )
            towrite = (
                " ".join(map(str, [int(self.xmin), int(self.ymin), float(angle)]))
                + "\n"
            )
            f.write(towrite)
            f.write(
                " ".join(self.AHE.round(2).astype(str).flatten(order="F")).replace(
                    ".0 ", " "
                )
            )
        return filename
