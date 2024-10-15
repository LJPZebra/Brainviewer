import napari
import numpy as np
from napari.utils.notifications import show_error, show_info, show_warning

from .colormaps import map_color


class PointLayerPairwise:
    def __init__(
        self,
        layer,
        pairwise,
        cmap,
        crange,
    ):
        self.layer = layer
        assert isinstance(layer, napari.layers.points.points.Points)

        # checking if pairwise matrix has the correct nb of elements
        self.pairwise = pairwise
        assert self.pairwise.shape[0] == self.pairwise.shape[1]
        assert len(self.layer.data) == self.pairwise.shape[0]

        # handling colors
        self.cmap = cmap
        self.crange = crange

        @self.layer.mouse_drag_callbacks.append
        def click_finder(l, e):
            if e.button != 2:
                return
            pos = l.world_to_data(e.position)
            dists = np.sum((layer.data - pos) ** 2, axis=1)
            i = np.argmin(dists)
            change_point_colors(
                l, i, map_color(self.cmap, self.pairwise[i], self.crange)
            )

        show_info("Right Click on a neuron to display Pairwise.")


def change_point_colors(layer, i, colors):
    colors[i, :] = [1, 0, 0, 1]
    layer.face_color = colors

