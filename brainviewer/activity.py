import matplotlib.pyplot as plt
import napari
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from napari.utils.notifications import show_error, show_info, show_warning


class ActivityViewer:
    def __init__(self, brain_viewer, slider_link=False):
        self.nbv = brain_viewer
        self.v = self.nbv._viewer

        plt.style.use("dark_background")
        self.canvas = FigureCanvas(Figure(tight_layout=True, frameon=False))
        self.ax = self.canvas.figure.subplots()

        self.widget = self.v.window.add_dock_widget(
            self.canvas, area="bottom", name="Activity"
        )

        if slider_link:  # if we want to draw a vertical bar at current time frame
            t_line = self.ax.axvline(0)

            @self.v.dims.events.current_step.connect
            def time_slider(e):
                t = self.v.dims.current_step[0]
                t_line.set_xdata([t])
                t_line.figure.canvas.draw()

    def update(self):
        self.ax.figure.canvas.draw()

    def clear(self):
        self.ax.clear()
        self.update()

    def rescale_y(self):
        mins = [l.get_ydata().min() for l in self.ax.lines]
        maxs = [l.get_ydata().max() for l in self.ax.lines]
        mmin = np.min(mins)
        mmax = np.max(maxs)
        self.ax.set_ylim(mmin, mmax)
        self.update()


class PointLayerSelector:
    def __init__(self, layer, widget, activities, labels=None):
        self.layer = layer
        assert isinstance(layer, napari.layers.points.points.Points)
        self.widget = widget

        if isinstance(activities, list):
            self.activities = activities
        elif isinstance(activities, np.ndarray):
            self.activities = [activities]
        else:
            raise TypeError()

        if labels is None:
            self.labels = None
        else:
            assert len(labels) == len(self.activities)
            self.labels = labels

        # checking if activities have the correct nb of elements
        for a in self.activities:
            assert len(self.layer.data) == a.shape[1]

        # prepare lines for the plot
        self.lines = []
        for a in self.activities:
            (line,) = self.widget.ax.plot(a[:, 0])
            self.lines.append(line)
        change_point_colors(self.layer, 0)

        @self.layer.mouse_drag_callbacks.append
        def click_finder(l, e):
            if e.button != 2:
                return
            pos = l.world_to_data(e.position)
            dists = np.sum((layer.data - pos) ** 2, axis=1)
            i = np.argmin(dists)
            change_point_colors(l, i)
            for j in range(len(self.activities)):
                self.lines[j].set_ydata(self.activities[j][:, i])
            self.widget.rescale_y()

        show_info("Right Click on a neuron to display its activity.")


def change_point_colors(layer, i):
    colors = np.ones((layer.data.shape[0], 4))
    colors[i, :] = [1, 0, 0, 1]
    layer.face_color = colors
