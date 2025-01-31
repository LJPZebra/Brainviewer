from pathlib import Path

import napari
import numpy as np
import zarr
from napari.utils.notifications import show_info, show_warning
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
        # self._set_dimensions()

    def _set_dimensions(self, ndims=4):
        d = self._viewer.dims
        if ndims == 3:
            d.ndim = 3
            d.axis_labels = ["t", "x", "y"]
            d.order = [0, 1, 2]
        elif ndims == 4:
            d.ndim = 4
            d.axis_labels = ["z", "t", "x", "y"]
            d.order = [1, 0, 2, 3]
        else:
            raise NotImplementedError()

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
        # assert coords.shape[1] == 3

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
                # units=(self.space_unit,) * 3,
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

    def contours(self, contours, **kwargs):
        layer = self._viewer.add_shapes(name="contours")
        colors = np.random.rand(len(contours), 3)
        ids = []
        for i in range(len(contours)):
            contour = contours[i]
            for c in contour:
                layer.add_polygons(
                    c,
                    face_color=colors[i],
                    edge_width=0,
                    **kwargs,
                )
                ids.append(i)

        layer.ids = np.array(ids)
        return layer

    def image(self, img_arr, cmap="inferno", clims=None, **kwargs):
        assert isinstance(img_arr, np.ndarray)
        assert img_arr.ndim == 2

        if clims is None:
            clims = np.quantile(img_arr, [0.05, 0.95])

        layer = self._viewer.add_image(
            img_arr,
            contrast_limits=clims,
            colormap=cmap,
            **kwargs,
        )
        return layer

    def stack(self, stack_arr, cmap="inferno", clims=None, **kwargs):
        assert isinstance(stack_arr, np.ndarray)
        assert stack_arr.ndim == 3
        if clims is None:
            clims = np.quantile(
                stack_arr[stack_arr.shape[0] // 2, :, :],
                [0.5, 0.9999],
            )
        layer = self._viewer.add_image(
            stack_arr,
            contrast_limits=clims,
            multiscale=False,
            colormap=cmap,
            **kwargs,
        )
        return layer

    def hyperstack(self, hstack_arr, cmap="inferno", **kwargs):
        assert isinstance(hstack_arr, (np.ndarray, zarr.core.Array))
        assert hstack_arr.ndim == 4
        contrast_limits = np.quantile(
            hstack_arr[hstack_arr.shape[0] // 2, hstack_arr.shape[1] // 2, :, :],
            [0.5, 0.9999],
        )
        layer = self._viewer.add_image(
            hstack_arr,
            contrast_limits=contrast_limits,
            multiscale=False,
            colormap=cmap,
            **kwargs,
        )
        self._set_dimensions()
        return layer

    def load_nrrd(self, path=None, cmap="magenta", **kwargs):
        import nrrd

        if path is None:
            path = self._ui_select_file("Select .nrrd")

        imgs, header = nrrd.read(path)
        px_size = header["space directions"][np.diag_indices(3)]  # [::-1]
        return self._viewer.add_image(imgs, scale=px_size, colormap=cmap, **kwargs)

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

    def select_rect_ROI(self, width=150, height=150):
        from qtpy.QtWidgets import QPushButton

        # preparing default ROI
        # size = self._viewer.dims.nsteps
        # xmid, ymid = size[2] // 2, size[3] // 2
        # xmin, xmax = xmid - width // 2, xmid + width // 2
        # ymin, ymax = ymid - height // 2, ymid + height // 2
        xmin, xmax = 0, width
        ymin, ymax = 0, height
        rectangle = np.array(
            [
                [xmin, ymin],
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
            ]
        )

        # creating ROI layer
        roi_layer = self._viewer.add_shapes(
            rectangle,
            name="ROI",
            edge_color="#fff",
            edge_width=5,
            face_color=[0, 0, 0, 0],
            # scale=px,
        )
        roi_layer.mode = "select"

        # creating UI
        qtbutton = QPushButton("Select Area")
        button = self._viewer.window.add_dock_widget(qtbutton, area="left")

        def clicked():
            if roi_layer.nshapes != 1:
                show_warning("Only 1 rectangle allowed !")
            if roi_layer.shape_type[0] != "rectangle":
                show_warning("Only rectangle shape allowed !")
            button.close()
            show_info("gathering ROI to NapariBrainViewer._ROI")
            roi_layer.mode = "pan_zoom"
            xyROI = np.sort(roi_layer.corner_pixels, axis=0)
            # z = self._viewer.dims.current_step[0]
            # self._ROI = tuple(xyROI.flatten()) + (z,)
            self._ROI = {
                "type": "rectangle",
                "x": xyROI[:, 0],
                "y": xyROI[:, 1],
                "z": self._viewer.dims.current_step[0],
                "t": self._viewer.dims.current_step[1],
            }

        qtbutton.clicked.connect(clicked)

    def apply_region_to_hyperstack(self, layer=None, roi=None):
        if roi is None:
            if hasattr(self, "_ROI"):
                roi = self._ROI
            else:
                raise TypeError()

        if layer is None:
            layer = self._viewer.layers.selection.active
            hyperstack = layer.data

        if roi["type"] == "rectangle":
            # xmin, ymin, xmax, ymax, z = roi
            region = hyperstack[
                roi["z"],
                :,
                roi["x"][0] : roi["x"][1],
                roi["y"][0] : roi["y"][1],
            ]
            return region

        else:
            raise NotImplementedError()
