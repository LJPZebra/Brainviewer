from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import QFileDialog

from .hdf5_handling import HDF5_Browser
from .selection import SelectionTab


class NapariBrainViewer:
    def __init__(self, work_dir=None, space_unit="μm", time_unit="sec"):
        self._selection_dock = None
        self._viewer = napari.Viewer()
        self.work_dir = work_dir
        self.space_unit = space_unit
        self.time_unit = time_unit

        # custom menu
        self._brain_menu = self._viewer.window.main_menu.addMenu("&BrainTools")

        # data loading sub-menu
        data_menu = self._brain_menu.addMenu("&Load data")
        data_menu.addAction("NRRD (.nrrd)", self.load_nrrd)
        data_menu.addAction("Zarr (.zarr)", self.load_zarr)
        data_menu.addAction("HDF5 (.h5)", self.load_hdf5)

        # setting up the viewer
        self._set_dimensions()


    def _set_dimensions(self):
        d = self._viewer.dims
        d.ndim = 4
        d.axis_labels = ["t", "z", "x", "y"]

    def _ui_select_files(self, prompt):
        files = self._viewer.window._qt_viewer._open_file_dialog_uni(prompt)
        return [Path(p) for p in files]

    def _ui_select_file(self, prompt):
        paths = self._ui_select_files(prompt)
        assert len(paths) == 1
        return paths[0]

    def _ui_select_directory(self, prompt):
        dlg = QFileDialog()
        path = dlg.getExistingDirectory(self._viewer.window._qt_viewer, prompt)
        print(path)
        return Path(path)

    @property
    def viewer(self):
        return self._viewer

    @property
    def brain_menu(self):
        return self._brain_menu

    def close(self):
        self._viewer.close()

    def points(
        self,
        coords,
        values=None,
        cmap="inferno",
        crange=None,
        size=None,
        **kwargs,
    ):
        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 3

        if size is None:
            # if self.pixel_size is not None:
            #     size = 8 / self.pixel_size[-1]
            # else:
            size = 8

        if values is None:
            layer = self._viewer.add_points(
                coords,
                size=size,
                out_of_slice_display=True,
                units=(self.space_unit,) * 3,
                **kwargs,
            )
        else:
            assert len(values) == len(coords)
            features = {"val": values}
            layer = self._viewer.add_points(
                coords,
                size=size,
                out_of_slice_display=True,
                units=(self.space_unit,) * 3,
                face_color="val",
                face_colormap=cmap,
                face_contrast_limits=crange,
                border_width=0,
                features=features,
                **kwargs,
            )

        return layer

    def load_nrrd(self, path=None, cmap="magenta", **kwargs):
        import nrrd

        if path is None:
            path = self._ui_select_file("Select .nrrd")

        imgs, header = nrrd.read(path)
        px_size = header["space directions"][np.diag_indices(3)][::-1]
        return self._viewer.add_image(imgs.T, scale=px_size, colormap=cmap, **kwargs)

    def load_zarr(self, path=None, cmap="magenta"):
        import zarr

        if path is None:
            path = self._ui_select_directory("Select .zarr directory")
            print(path)
        return NotImplementedError()
        imgs = zarr.open(path, "r")
        # imgs, header = nrrd.read(path)
        # px_size = header["space directions"][np.diag_indices(3)][::-1]
        # return self._v.add_image(imgs.T, scale=px_size, colormap=cmap)
        return imgs

    def load_hdf5(self, path=None):
        if path is not None:
            raise NotImplementedError()

        self._h5widget = self._viewer.window.add_dock_widget(HDF5_Browser(self))
