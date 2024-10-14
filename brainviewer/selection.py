import napari
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from dask.array import shape
from napari.utils.notifications import show_info

from .colormaps import map_color

def _log(msg):
    with open("selections_log.txt", 'a') as f:
        f.write(f'-> {msg}\n\n')

def is_in_rectangle(vertices, point):
    x, y = point
    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    return x_min <= x <= x_max and y_min <= y <= y_max

def region_pairing(pairing_matrix, region):
    """
    Pairing matrix is a square matrix of size n x n, where n is the number of neurons.
    The region is a list of indices of neurons that are in the region.
    """
    # _log(f'pairing_matrix.shape = {pairing_matrix.shape}; region = {region}')
    # return np.zeros(pairing_matrix.shape[0])-1

    return np.abs(pairing_matrix[:,region]).sum(axis=1) / region.size

def change_point_colors(layer, highlighted_points, values, cmap, crange):
    _log(f'highlighted_points = {highlighted_points}')
    colors = map_color(cmap, values, crange)
    colors[highlighted_points, :] = [1, 1, 1, 1]
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
        self._selection = PointsSelection(brain_viewer, points_layer, matrix, cmap, crange)

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
            face_color='blue',
            opacity=0.5,
            name="Rectangle selection layer",
        )
        shape_layer.mode = 'add_rectangle'

        def on_shape_change():
            self._selection.unselect_all()

            _log(f'shape layer data: {shape_layer.data}')

            for rectangle in shape_layer.data:
                self._selection.select_rectangle(rectangle)

        shape_layer.events.data.connect(on_shape_change)
        self._shape_layers.append(shape_layer)



class PointsSelection:
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


    def points_in_rectangle_selection(self, rectangle_vertices):
        points = self._points_layer.data
        selection = []

        if self._brain_viewer.viewer.dims.ndisplay == 3:
            return selection

        mask_points_in_slice = np.isclose(points[:, 0], self._brain_viewer.viewer.dims.point[0], atol=1.5)
        points = points[:, 1:]

        for i, point in enumerate(points):
            if is_in_rectangle(rectangle_vertices, point) and mask_points_in_slice[i]:
                selection.append(i)

        return selection


    def select_rectangle(self, rectangle_vertices):
        _log(f'select {rectangle_vertices}')

        self._selection.extend(self.points_in_rectangle_selection(rectangle_vertices))
        self.update_selection()


    def unselect_rectangle(self, rectangle_vertices):
        # return
        selection = self.points_in_rectangle_selection(rectangle_vertices)
        for i in selection:
            self._selection.remove(i)
        self.update_selection()


    def unselect_all(self):
        _log('unselect all')

        self._selection = []
        self.update_selection()


    def update_selection(self):

        _log(f'update: selection = {self.selection}')

        values = region_pairing(self._pairing_matrix, self.selection) if self.selection.size > 0 else 0
        # values=np.zeros(self._pairing_matrix.shape[0])-1
        change_point_colors(self._points_layer, self._selection, values, self._cmap, self._crange)