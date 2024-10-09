import numpy as np
from matplotlib import colormaps as mcm
from napari.utils import Colormap


def alpha_sigmoid(cm, x0=0.5, sharpness=20, name="Custom"):
    """Add an alpha channel with a sigmoid shape.

    Parameters
    ==========
    cm : str or matplotlib colormap
        colormap to which we add an alpha chanel
    x0 : float [0.,1.]
        center of the sigmoid.
    sharpness : float
        sharpness of the sigmoid
    name : str
        name of the created colormap

    Return
    ======
    colormap : napari.utils.Colormap
        color map with added alpha channel

    Note
    ====
    the alpha channel will be of the shape :
    alpha = 1 - 1/(1+exp(sharpness*(x-x0)))
    with x in [0.,1.]
    """
    if isinstance(cm, str):
        cm = mcm[cm]

    if hasattr(cm, "colors"):
        colors = cm.colors
    else:
        colors = cm(np.linspace(0, 1, 256))

    x = np.linspace(0, 1, len(colors))
    alpha = 1 - 1 / (1 + np.exp(sharpness * (x - x0)))
    cols = np.c_[colors, alpha]
    return Colormap(cols, name=name, _display_name=name)


def alpha_cosine(cm, x0=0.5, sharpness=20, name="Custom"):
    """Add an alpha channel with a centered cosine shape.

    Parameters
    ==========
    cm : str or matplotlib colormap
        colormap to which we add an alpha chanel
    x0 : float [0.,1.]
        center of the cosine.
    sharpness : float
        sharpness of the cosine
    name : str
        name of the created colormap

    Return
    ======
    colormap : napari.utils.Colormap
        color map with added alpha channel

    Note
    ====
    the alpha channel will be of the shape :
    alpha = 1 - cos(pi*(x-x0))^sharpness
    with x in [0.,1.]
    """
    if isinstance(cm, str):
        cm = mcm[cm]

    if hasattr(cm, "colors"):
        colors = cm.colors
    else:
        colors = cm(np.linspace(0, 1, 256))
        colors = colors[:, :3]

    x = np.linspace(0, 1, len(colors))
    alpha = 1 - (np.cos(np.pi * (x - x0))) ** sharpness
    cols = np.c_[colors, alpha]
    return Colormap(cols, name=name, _display_name=name)


def map_color(colormap: Colormap, values: np.ndarray, contrast_limits: tuple):
    normalized = (values - contrast_limits[0]) / (
        contrast_limits[1] - contrast_limits[0]
    )
    colors = colormap.map(normalized)
    return colors


cm_inferno_alpha = alpha_sigmoid("inferno", 0.5, 10, "InfernoAlpha")
cm_seismic_alpha = alpha_cosine("seismic", 0.5, 20, "SeismicAlpha")
