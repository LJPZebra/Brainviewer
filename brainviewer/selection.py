import napari
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from dask.array import shape
from napari.utils.notifications import show_info
from skimage.data import brain

from .colormaps import map_color


def enable_selection(
        brain_viewer,
        points_layer,
        matrix,
        cmap,
        crange,
    ):
    # callback
    def show_selection_tab():
        brain_viewer.viewer.window.add_dock_widget(
            SelectionTab(brain_viewer, points_layer, matrix, cmap, crange),
            name="Selection",
        )
    # menu entry
    brain_viewer.brain_menu.addAction("&Selection tab", show_selection_tab)


def region_pairing(pairing_matrix, region):
    """
    Pairing matrix is a square matrix of size n x n, where n is the number of neurons.
    The region is a list of indices of neurons that are in the region.
    """
    # _log(f'pairing_matrix.shape = {pairing_matrix.shape}; region = {region}')
    # return np.zeros(pairing_matrix.shape[0])-1

    return np.abs(pairing_matrix[:,region]).sum(axis=1) / region.size


def is_in_rectangle(vertices, point):
    x, y = point
    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    return x_min <= x <= x_max and y_min <= y <= y_max

def change_point_colors(layer, highlighted_points, values, cmap, crange):
    colors = map_color(cmap, values, crange)
    colors[highlighted_points, :] = [.5, .5, 1, 1]
    layer.face_color = colors



class SelectionTab(QWidget):
    def __init__(
            self,
            brain_viewer,
            points_layer,
            matrix,
            cmap,
            crange,
    ):
        super().__init__()
        self._brain_viewer = brain_viewer
        self._points_layer = points_layer
        self._matrix = matrix
        self._cmap = cmap
        self._crange = crange
        self._shape_layers = []
        self._selection_layer = SelectionLayer(brain_viewer, points_layer, matrix, cmap, crange)

        self.setLayout(QVBoxLayout())

        new_selection_button = QPushButton("Create selection layer")
        new_selection_button.clicked.connect(self.add_selection_layer)
        self.layout().addWidget(new_selection_button)


    def add_selection_layer(self):
        shape_layer = self._brain_viewer.viewer.add_shapes(
            data=None,
            shape_type='rectangle',
            edge_width=1,
            edge_color='red',
            face_color='#ffffff3f',
            opacity=0.5,
            name="Rectangle selection layer",
        )
        shape_layer.mode = 'add_rectangle'

        def on_shape_change():
            self._selection_layer.unselect_all()

            for rectangle in shape_layer.data:
                self._selection_layer.select_rectangle(rectangle)

        shape_layer.events.data.connect(on_shape_change)
        self._shape_layers.append(shape_layer)


class SelectionLayer:
    def __init__(self, brain_viewer, points_layer, pairing_matrix, cmap, crange):

        assert isinstance(points_layer, napari.layers.points.points.Points)
        self._brain_viewer = brain_viewer
        self._points_layer = points_layer
        self._pairing_matrix = pairing_matrix
        self._cmap = cmap
        self._crange = crange
        self._selection = []


    @property
    def selection(self):
        return np.array(self._selection)


    def points_in_rectangle_selection(self, rectangle_vertices, thickness=1.5):
        points = self._points_layer.data
        selection = []

        if self._brain_viewer.viewer.dims.ndisplay == 3:
            return selection

        mask_points_in_slice = np.isclose(points[:, 0], self._brain_viewer.viewer.dims.point[0], atol=thickness/2)
        points = points[:, 1:]

        for i, point in enumerate(points):
            if is_in_rectangle(rectangle_vertices, point) and mask_points_in_slice[i]:
                selection.append(i)

        return selection


    def select_rectangle(self, rectangle_vertices):
        self._selection.extend(self.points_in_rectangle_selection(rectangle_vertices))
        self.update_selection()


    def unselect_rectangle(self, rectangle_vertices):
        selection = self.points_in_rectangle_selection(rectangle_vertices)
        for i in selection:
            self._selection.remove(i)
        self.update_selection()


    def unselect_all(self):
        self._selection = []
        self.update_selection()


    def update_selection(self):
        values = region_pairing(self._pairing_matrix, self.selection) if self.selection.size > 0 else 0
        change_point_colors(self._points_layer, self._selection, values, self._cmap, self._crange)