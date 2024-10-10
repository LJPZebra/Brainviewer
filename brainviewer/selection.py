import napari
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .colormaps import map_color

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
    return pairing_matrix[:,region].abs().sum(axis=1) / len(region)

def change_point_colors(layer, initial_points, values, cmap, crange):
    colors = map_color(cmap, values, crange)
    colors[initial_points, :] = [1, 0, 0, 1]
    layer.face_color = colors

class SelectionTab(QWidget):
    def __init__(
            self,
            brain_viewer,
            points_layer,
            slice_z,
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
        self._selection = PointsSelection(points_layer, slice_z, matrix)

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
        shape_layer.events.data.connect(self.on_shape_change)
        self._shape_layers.append(shape_layer)

    def on_shape_change(self):
        shape_layer = self._shape_layers[-1]
        with open("selections_log.txt", "a") as f:
            for shape in shape_layer.data:
                f.write(f"Shape coordinates: {shape}\n")
            f.write("---\n")



class PointsSelection:
    def __init__(self, points_layer, slice_z, pairing_matrix, cmap, crange):

        assert isinstance(points_layer, napari.layers.points.points.Points)
        self._points_layer = points_layer
        self._slice = slice_z
        self._pairing_matrix = pairing_matrix
        self._cmap = cmap
        self._crange = crange
        self._selection = []

    def points_in_rectangle_selection(self, rectangle_vertices):
        points = self._points_layer.data

        if points.shape[1] == 3:
            return

        selection = []
        for i, point in enumerate(points):
            if is_in_rectangle(rectangle_vertices, point):
                self._selection.append(i)
        return selection

    def select_rectangle(self, rectangle_vertices):
        self._selection.append(self.points_in_rectangle_selection(rectangle_vertices))
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
        values = region_pairing(self._pairing_matrix, self._selection)
        change_point_colors(self._points_layer, self._selection, values, self._cmap, self._crange)