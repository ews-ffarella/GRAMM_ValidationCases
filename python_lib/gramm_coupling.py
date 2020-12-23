import io
import pandas as pd
import geopandas as gpd
import numpy as np
import fiona as fio
import rasterio as rio
import shapely.geometry as sg
from pathlib import Path
import subprocess
import rasterio.plot
import matplotlib.pyplot as plt
import rasterio.plot
import descartes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import struct
import vtk
import rasterio.coords
import rasterio.transform
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

def readint(stream):
    return struct.unpack("i", stream.read(4))[0]


def readfloat(stream):
    return struct.unpack("f", stream.read(4))[0]


def readint(stream):
    return struct.unpack("i", stream.read(4))[0]

def readfloat(stream):
    return struct.unpack("f", stream.read(4))[0]

def readdouble(stream):
    return struct.unpack("d", stream.read(8))[0]


def read_gramm_grid(filename):

    with filename.open("rb") as fin:
        _ = fin.read(6)  # header
        nx = readint(fin)
        ny = readint(fin)
        nz = readint(fin)
        slab_shape = (nx, ny)
        slab2d = nx * ny        
        ah = np.fromfile(fin, dtype="f", count=slab2d, sep="")         
        zsp = np.fromfile(fin, dtype="f", count=nx*ny*nz, sep="")
        x_pts = np.fromfile(fin, dtype="f", count=nx+1, sep="")      
        y_pts = np.fromfile(fin, dtype="f", count=ny+1, sep="")      
        z_pts = np.fromfile(fin, dtype="f", count=nz+1, sep="") 
        vol = np.fromfile(fin, dtype="f", count=nx * ny * nz, sep="")  
        areax = np.fromfile(fin, dtype="f", count=(nx+1) * ny * nz, sep="")  
        areay = np.fromfile(fin, dtype="f", count=nx * (ny+1) * nz, sep="")   
        areazx = np.fromfile(fin, dtype="f", count=nx * ny * (nz+1), sep="")  
        areazy = np.fromfile(fin, dtype="f", count=nx * ny * (nz+1), sep="") 
        areaz = np.fromfile(fin, dtype="f", count=nx * ny * (nz+1), sep="")     
        ddx = np.fromfile(fin, dtype="f", count=nx, sep="") 
        ddy = np.fromfile(fin, dtype="f", count=ny, sep="") 
        zax = np.fromfile(fin, dtype="f", count=nx, sep="") 
        zay = np.fromfile(fin, dtype="f", count=ny, sep="") 
        xmin = readint(fin)
        ymin = readint(fin)
        angle = readdouble(fin)
        z_pts_list = list()
        for _ in range(nz+1):
            z = np.fromfile(fin, dtype="f", count=(nx + 1) * (ny + 1), sep="")
            z = np.transpose(z.reshape((nx + 1, ny + 1), order="F"), [1, 0])
            z_pts_list.append(z)
        rest = np.fromfile(fin, dtype="f", sep="") 
       
    assert rest.size == 0, rest   
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions([nx+1, ny+1, nz+1])
    xs = xmin + np.append([0], ddx).cumsum().round(2)
    ys = ymin + np.append([0], ddy).cumsum().round(2)
    points = vtk.vtkPoints()
    XX, YY = np.meshgrid(xs, ys)
    xx = XX.flatten()
    yy = YY.flatten()
    arr = np.empty(((nx+1)*(ny+1)*(nz+1), 3))
    arr[:, 0] = np.concatenate([xx for arr in z_pts_list])
    arr[:, 1] = np.concatenate([yy for arr in z_pts_list])
    arr[:, 2] = np.concatenate([arr.flatten() for arr in z_pts_list])
    points.SetData(numpy_to_vtk(arr, deep=True))
    sgrid.SetPoints(points)
    return sgrid, (xs, ys, ah)
        


def numpy_array_to_vtk_scalars(array, name=None):
    nz = array.shape[2]
    scalars = numpy_to_vtk(np.concatenate([array[:, :, i].flatten() for i in range(nz)]), deep=True)
    if isinstance(name, (str,)):
        scalars.SetName(name)
    return scalars


def numpy_array3d_to_vtk(array, name=None):
    nx, ny, nz, n_components = array.shape
    assert n_components == 3
    size = nx * ny * nz
    arr_3d = numpy_to_vtk(
        np.concatenate([array[:, :, i, :].flatten() for i in range(nz)]).reshape((size, 3)), deep=True
    )
    if isinstance(name, (str,)):
        arr_3d.SetName(name)
    return arr_3d

class GrammMesh:
    def __init__(self, path, epsg_code=3416):
        path = Path(path)
        assert path.exists(), f"Project path {path!r} does not exist"
        self.path = path
        self.epsg_code = None
        self.r_crs = None
        self.crs = None
        self.projection_name = None
        self.vertical_expension_ratio = None
        self.terrain_cell_height = None
        self.n_border_smooth_cells = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dx = None
        self.dy = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.vtk_grid = self.terrain_data = None
        self.read_grid_dimensions()
        if isinstance(epsg_code, (int, float)):
            self.setup_crs(epsg_code)
        
    def setup_crs(self, epsg_code):
        self.epsg_code = int(epsg_code)
        self.r_crs = rio.crs.CRS.from_epsg(self.epsg_code)
        assert self.r_crs.is_projected
        assert self.r_crs.linear_units == "metre"
        self.crs = self.r_crs.to_dict()
        self.projection_name = self.r_crs.to_wkt().split('"')[1].split('"')[0]

    @property
    def domain_width(self):
        return self.xmax - self.xmin

    @property
    def domain_height(self):
        return self.ymax - self.ymin

    @property
    def extent(self):
        return [
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        ]

    @property
    def bounds(self):
        return rio.coords.BoundingBox(self.xmin, self.ymin, self.xmax, self.ymax,)

    @property
    def transform(self):
        return rio.transform.from_bounds(
            self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top, self.nx, self.ny,
        )

    @property
    def nx_points(self):
        return self.nx + 1

    @property
    def ny_points(self):
        return self.ny + 1

    @property
    def nz_points(self):
        return self.nz + 1

    @property
    def number_of_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def number_of_points(self):
        return self.nx_points * self.ny_points * self.nz_points
        
    def read_grid_dimensions(self):
        fname = self.path / 'geom.in'
        assert fname.is_file()
        data = fname.read_text().split('\n')
        self.terrain_cell_height = float(data[1])
        self.vertical_expension_ratio = float(data[2])
        try:
            self.n_border_smooth_cells = int(data[3])
        except:
            self.n_border_smooth_cells = 0
        fname = self.path / "GRAMM.geb"
        assert fname.is_file()
        nx, ny, nz, xmin, xmax, ymin, ymax = map(
            float, [_.split("!")[0].strip() for _ in fname.read_text().split("\n") if _]
        )
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        
    def read_mesh(self):
        fname = self.path / "ggeom.asc"
        assert fname.is_file()
        sgrid, (xs, ys, ah) = read_gramm_grid(fname)
        self.xs = xs
        self.ys = ys
        nxp, nyp, nzp = sgrid.GetDimensions()
        assert (nxp-1) == self.nx
        assert (nyp-1) == self.ny
        assert (nzp-1) == self.nz
        
        extents = sgrid.GetExtent()
        subgrid = vtk.vtkExtractGrid()
        subgrid.SetVOI((extents[0], extents[1], extents[2], extents[3], extents[4], 0))
        subgrid.SetIncludeBoundary(True)
        subgrid.SetInputData(sgrid)
        subgrid.Update()
        terrain_data = vtk.vtkStructuredGrid()
        terrain_data.ShallowCopy(subgrid.GetOutput())
        subgrid = None
        del subgrid        

        ah = np.transpose(ah.reshape((nxp-1, nyp-1), order="F"), [1, 0])     
        ah = numpy_to_vtk(ah.flatten(), deep=True)
        ah.SetName("Elevation")
        terrain_data.GetCellData().SetScalars(ah)
        
        # terrain_data.GetPointData().SetScalars(elevation)
        
        #src/Gral/GRALIO/Landuse.cs
        '''
        //Albedo [%]
        double [] AGL=new double[1000];
        //Emissivity of the surface [%]
        double[] EPSGL = new double[1000];
        //Soil moisture [%]
        double[] FWL = new double[1000];
        //Roughness length [m]
        double[] Z0L = new double[1000];
        //Heat conductivity [W/m" + SquareString]
        double[] ALAMBDAL = new double[1000];
        //Temperature conductivity [m�/s]
        double[] ALAMBDAT = new double[1000];
        '''
        
        self.terrain_data = terrain_data
        self.vtk_grid = sgrid
                                 
        # RHOB ALAMBDA Z0 FW EPSG ALBEDO
        if not self.path.joinpath('Landuse.asc').is_file():
            return self.vtk_grid 
        
        try:
            arr = self.path.joinpath('Landuse.asc').read_text().strip().split("\n")
            arr = [np.array(_.split(' ')).astype('float') for _ in arr]
           
            nt = self.nx * self.ny 
            i = 0
            for ai, name in enumerate('Density Conductivity RoughnessLength MoistureContent Emissivity Albedo'.split(' ')):
                avtk = arr[i]
                i += 1  
                avtk = np.transpose(avtk.reshape((nxp-1, nyp-1), order="F"), [1, 0])       
                avtk = numpy_to_vtk(avtk.flatten(), deep=True)
                avtk.SetName(name)
                terrain_data.GetCellData().AddArray(avtk)    
        except Exception as e:
            print(e)
            pass

        return self.vtk_grid 

    @property
    def cell_heights(self):
        return self.terrain_cell_height * self.vertical_expension_ratio ** np.arange(self.nz)

    @property
    def cell_tops(self):
        return self.cell_heights.cumsum()


    def write_grid(self, fname=None, grid=None):
        if grid is None:
            grid = self.vtk_grid
        if fname is None:
            fname = self.path / 'mesh.vts'
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(str(fname))
        writer.SetInputData(grid)
        res = writer.Write()
        writer = None
        del writer
        return res

    def write_terrain_data(self, fname=None, grid=None):
        if fname is None:
            fname = self.path / 'terrain_data.vts'
        if grid is None:
            grid = self.terrain_data
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(str(fname))
        writer.SetInputData(grid)
        res = writer.Write()
        writer = None
        del writer
        return res

    def plot_terrain_data(self, array_name, location_gdf=None):
        resolution = self.dx
        arr = self.terrain_data.GetCellData().GetArray(array_name)
        n_border_smooth_cells = self.n_border_smooth_cells
        buffer_distance = n_border_smooth_cells * resolution
        bounds = self.bounds    
        extent = self.extent
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        arr = vtk_to_numpy(arr).reshape(self.nx, self.ny, order = 'F')
        im = ax.pcolormesh(
            self.xs, self.ys, arr.T,
            cmap='terrain', zorder=20
        )
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal');

        if location_gdf is not None:
            ax.plot(
                [pt.x for pt in location_gdf.geometry], 
                [pt.y for pt in location_gdf.geometry], 
                zorder=50, ms=5, ls='none',  marker='o', mec='k', mfc='white')

        if buffer_distance > 0:     
            ax.add_patch(
                descartes.patch.PolygonPatch(sg.box(*bounds).buffer(-buffer_distance), fc='none', ec='k', lw=2, zorder=100, ls='--')
            );

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, orientation='vertical', cax=cax)
        ax.grid(zorder=10000, which='major', visible=True, axis='both', color='k')
        ax.xaxis.set_major_locator(plt.MultipleLocator(10*resolution))
        ax.yaxis.set_major_locator(plt.MultipleLocator(10*resolution))
        ax.tick_params(rotation=90, axis='x')
        fig.tight_layout()
        return fig, ax, cax

    def plot_slab(self, marr, z_level, location_gdf=None, cmap='viridis'):
        resolution = self.dx
        arr = marr[:, :, z_level].reshape(self.ny, self.nx, order = 'F')[::-1, :]
        n_border_smooth_cells = self.n_border_smooth_cells
        buffer_distance = n_border_smooth_cells * resolution
        bounds = self.bounds    
        extent = self.extent
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.pcolormesh(
            self.xs, self.ys, arr,
            cmap=cmap, zorder=20
        )
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal');

        if location_gdf is not None:
            ax.plot(
                [pt.x for pt in location_gdf.geometry], 
                [pt.y for pt in location_gdf.geometry], 
                zorder=50, ms=5, ls='none',  marker='o', mec='k', mfc='white')

        if buffer_distance > 0:     
            ax.add_patch(
                descartes.patch.PolygonPatch(sg.box(*bounds).buffer(-buffer_distance), fc='none', ec='k', lw=2, zorder=100, ls='--')
            );

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, orientation='vertical', cax=cax)
        ax.grid(zorder=10000, which='major', visible=True, axis='both', color='k')
        ax.xaxis.set_major_locator(plt.MultipleLocator(10*resolution))
        ax.yaxis.set_major_locator(plt.MultipleLocator(10*resolution))
        ax.tick_params(rotation=90, axis='x')
        fig.tight_layout()
        return fig, ax, cax

    def add_array_to_mesh(self, array, array_name):
        cell_data = self.vtk_grid.GetCellData()
        is_3d = False
        if len(array.shape) == 4:        
            expected_shape = (self.ny, self.nx, self.nz, 3)
            assert array.shape == expected_shape, (array.shape, expected_shape)
            is_3d = True
        elif len(array.shape) == 3:        
            expected_shape = (self.ny, self.nx, self.nz)
            assert array.shape == expected_shape, (array.shape, expected_shape)
        else:
            raise RunTimeError("Wrong shape")
        if not is_3d:
            cell_data.AddArray(numpy_array_to_vtk_scalars(array, name=array_name))  
        else:
            cell_data.AddArray(numpy_array3d_to_vtk(array, name=array_name))     
        return array_name

    @classmethod
    def extract_cell_centers(cls, vtk_grid):    
        nx, ny, nz = vtk_grid.GetDimensions()
        nx -= 1
        ny -= 1
        nz -= 1
        cellCenters = vtk.vtkCellCenters()
        cellCenters.SetInputData(vtk_grid)
        cellCenters.Update()
        cc = vtk_to_numpy(cellCenters.GetOutput().GetPoints().GetData())
        cc = cc.reshape(nz, ny, nx, 3, order = 'C')
        cc = np.transpose(cc, [3, 1, 2, 0])
        writer = cellCenters
        del cellCenters
        return cc
    
    @classmethod
    def find_vtk_terrain_heights(cls, coords2D, terrain_mesh, step=1e-6, max_tries=10, no_data_value=-999):
        if isinstance(coords2D, pd.DataFrame):
            coords2D = coords2D.values
        elif isinstance(coords2D, list):
            coords2D = np.array(coords2D)
        assert isinstance(coords2D, np.ndarray), "Should be a numpy array"
        assert len(coords2D.shape) == 2, "Should be of shape (n_pts, 2)"
        assert coords2D.shape[1] == 2, "Should be of shape (n_pts, 2)"
        xmin, xmax, ymin, ymax, zmin, zmax = terrain_mesh.GetBounds()
        z_min = zmin - 50.0
        z_max = zmax + 50.0
        x_mid = 0.5 * (xmax + xmin)
        y_mid = 0.5 * (ymax + ymin)
        # We create the oriented bounding-box tree
        obbTree = vtk.vtkModifiedBSPTree()
        obbTree.SetDataSet(terrain_mesh)
        obbTree.BuildLocator()
        tolerance = 0.00001
        if coords2D.shape[-1] != 2:
            raise RuntimeError
        # coords_shp = coords2D.shape[:-1]
        n_points = int(coords2D.size / 2.0)
        # We create the points
        terrain_heights = np.empty((n_points, 2), dtype=np.float)
        for i, xy_coord in enumerate(coords2D):
            x_coord, y_coord = xy_coord
            pSource = [x_coord, y_coord, z_min]
            pTarget = [x_coord, y_coord, z_max]
            n_tries = 0
            ret = -1
            z_terrain = [np.nan, -1]
            while ret == -1:
                n_tries += 1
                if n_tries > max_tries:
                    break
                if pSource[0] <= x_mid:
                    pSource[0] += step
                    pTarget[0] += step
                else:
                    pSource[0] -= step
                    pTarget[0] -= step

                if pSource[1] <= y_mid:
                    pSource[1] += step
                    pTarget[1] += step
                else:
                    pSource[1] -= step
                    pTarget[1] -= step
                t = vtk.mutable(0)
                hit_coords = [0.0, 0.0, 0.0]
                pcoords = [0.0, 0.0, 0.0]
                subId = vtk.mutable(0)
                hit_cellId = vtk.mutable(0)
                hit_cell = vtk.vtkGenericCell()
                has_hit = obbTree.IntersectWithLine(
                    pSource, pTarget, tolerance, t, hit_coords, pcoords, subId, hit_cellId, hit_cell,
                )
                hit_cell_zMinMax = list(hit_cell.GetBounds()[-2:])
                if no_data_value in hit_cell_zMinMax:
                    z_terrain = [np.nan, -1]
                    ret = 1
                    break
                if (
                    abs(pSource[0] - hit_coords[0]) > tolerance
                    or abs(pSource[1] - hit_coords[1]) > tolerance
                    or has_hit != 1
                ):
                    z_terrain = [np.nan, -1]
                    ret = -1
                else:
                    if (z_terrain[0] < zmin) or (z_terrain[0] > zmax):
                        z_terrain = [np.nan, -1]
                        ret = -1
                    else:
                        z_terrain = [hit_coords[2], hit_cellId]
                        ret = 1
            terrain_heights[i] = z_terrain
        return terrain_heights[:, 0]

    @classmethod
    def extract_terrain_cells(cls, vtk_grid):
        extents = vtk_grid.GetExtent()
        subgrid = vtk.vtkExtractGrid()
        subgrid.SetVOI((extents[0], extents[1], extents[2], extents[3], extents[4], 0))
        subgrid.SetIncludeBoundary(True)
        subgrid.SetInputData(vtk_grid)
        subgrid.Update()
        res = vtk.vtkStructuredGrid()
        res.ShallowCopy(subgrid.GetOutput())
        subgrid = None
        del subgrid
        return res
    
    @classmethod
    def points3d_to_polydata(cls, pts_3d):
        if isinstance(pts_3d, pd.DataFrame):
            pts_3d = pts_3d.values
        elif isinstance(pts_3d, list):
            pts_3d = np.array(pts_3d)
        assert isinstance(pts_3d, np.ndarray), "Should be a numpy array"
        assert len(pts_3d.shape) == 2, "Should be of shape (n_pts, 3)"
        assert pts_3d.shape[1] == 3, "Should be of shape (n_pts, 3)"

        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(pts_3d))
        pdata = vtk.vtkPolyData()
        pdata.SetPoints(pts)
        assert pdata.GetPoints().GetNumberOfPoints() == pts_3d.shape[0]
        return pdata
    
    @classmethod
    def sample_poly_data(cls, data_to_interpolate, dataset, mask=True):
        probeFilter = vtk.vtkProbeFilter()
        probeFilter.SetInputData(data_to_interpolate)
        probeFilter.SetSourceData(dataset)
        probeFilter.Update()
        data = probeFilter.GetOutput().GetPointData()
        n_arrays = data.GetNumberOfArrays()
        array_names = [data.GetArrayName(i) for i in range(n_arrays)]
        arrays = [data.GetArray(i) for i in range(n_arrays)]
        array_n = [arr.GetNumberOfComponents() for arr in arrays]
        f = 1
        mask = mask and "vtkValidPointMask" in array_names
        if mask:
            vtkValidPointMask = vtk_to_numpy(data.GetArray("vtkValidPointMask"))
            f = np.ones(vtkValidPointMask.size)
            f[vtkValidPointMask == 0] = np.nan
        res = dict()
        for (name, arr, n) in zip(array_names, arrays, array_n):
            if name == "vtkValidPointMask":
                continue
            if n == 1:
                res[name] = vtk_to_numpy(arr) * f
                continue
            arr = vtk_to_numpy(arr)
            for i in range(n):
                res[f"{name}_{i}"] = arr[:, i] * f
        return res
    
def plot_probes(fname, n_probes, tail=None):
    fname = Path(fname)
    data_df = pd.read_csv(fname, skiprows=n_probes+3, sep=' ').set_index(['PROBEI', 'Time']).astype('float')
    end_loc = data_df.index.get_level_values('Time').max()
    data_df = data_df.where(data_df != -99999) 
    data_df['U'] = data_df.U / 1000.0
    data_df['V'] = data_df.V / 1000.0
    data_df['W'] = data_df.W / 10000.0
    data_df['T'] = data_df['T'] / 100.0
    data_df['K'] = data_df.K / 100000.0
    data_df['DP'] = data_df.DP / 1000.0
    data_df['EPSILON'] = data_df.EPSILON / 10000000.0
    data_df['U2D'] = ((data_df.U ** 2.0) + (data_df.V ** 2.0))**0.5
    data_df['INCL'] = np.degrees(np.arctan2(data_df.W, data_df.U2D))
    data_df['DIR'] = np.fmod(270 - np.degrees(np.arctan2(data_df.V, data_df.U)), 360.0)
    data_df['TI'] = 100.0 * ((4.0 * data_df.K / 3.0) ** 0.5) / data_df.U2D    
    start_loc = 0
    if tail is not None:
        start_loc = max(0, end_loc-tail)
    for c in ['U2D', 'INCL', 'DIR', 'TI', 'EPSILON', 'DP']:    
        data_df[c].unstack('PROBEI').sort_index().loc[start_loc:end_loc].plot(legend=False, title=c, figsize=(20, 5))
    return plt.gcf()
    

def read_wnd_file(fname, expected_shape=None, speed=True, direction=True):
    fname = Path(fname)
    if not fname.exists():
        return None, None, None
    _speed = _direction = None
    with fname.open("rb") as fin:
        _ = readint(fin)  # header
        nx = readint(fin)
        ny = readint(fin)
        nz = readint(fin)
        if expected_shape is not None:
            assert nx == expected_shape[0]
            assert ny == expected_shape[1]
            assert nz == expected_shape[2]
        _ = readfloat(fin)  # gridsize
        shape = (nx, ny, nz, 3)
        uvw = np.fromfile(fin, dtype="int16", count=nx * ny * nz * 3, sep="") / 100.0
        uvw = np.transpose(uvw.reshape(shape, order="C"), [1, 0, 2, 3])
    if speed:
        _speed = (uvw[:, :, :, 0] ** 2.0 + uvw[:, :, :, 1] ** 2.0) ** 0.5
    if direction:
        _direction = np.arctan2(uvw[:, :, :, 1], uvw[:, :, :, 0])
        _direction = np.fmod(270.0 - np.degrees(_direction), 360.0)
    return uvw, _speed, _direction

def read_ewnd_file(fname, expected_shape=None):
    fname = Path(fname)
    if not fname.exists():
        return None, None, None
    with fname.open("rb") as fin:
        _ = readint(fin)  # header
        nx = readint(fin)
        ny = readint(fin)
        nz = readint(fin)
        if expected_shape is not None:
            assert nx == expected_shape[0]
            assert ny == expected_shape[1]
            assert nz == expected_shape[2]
        _ = readfloat(fin)  # gridsize
        shape = (nx, ny, nz, 4)
        arr = np.fromfile(fin, dtype="int16", count=nx * ny * nz * 4, sep="")
        arr = np.transpose(arr.reshape(shape, order="C"), [1, 0, 2, 3])
    tke = arr[:, :, :, 0] / 10000.0
    epsilon = arr[:, :, :, 1] / 1000000.0
    dp = arr[:, :, :, 2] / 100.0
    t_abs= arr[:, :, :, 3] / 100.0
    
    return tke, epsilon, dp, t_abs



    