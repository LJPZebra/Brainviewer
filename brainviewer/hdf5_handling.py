from pathlib import Path

import h5py
from magicgui.widgets import FileEdit
from napari.utils.notifications import show_error, show_info
from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt
from qtpy.QtCore import QUrl
from qtpy.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                            QLabel, QLineEdit, QMenu, QPushButton, QTreeView,
                            QVBoxLayout, QWidget)


class HDF5TreeItem:
    """A simple tree item to hold information about each node."""

    def __init__(self, name, obj=None, parent=None):
        self.name = name
        self.obj = obj
        self.parent_item = parent
        self.child_items = []
        self.type = type(obj)

    def append_child(self, item):
        self.child_items.append(item)

    def child(self, row):
        return self.child_items[row]

    def child_count(self):
        return len(self.child_items)

    def column_count(self):
        return 1

    def data(self):
        if self.type == h5py.Group:
            return "📁" + self.name
        elif self.type == h5py.Dataset:
            return "🔢" + self.name + f"\t\t⚙️{self.obj.shape}{self.obj.dtype}"
        else:
            return self.name

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def parent(self):
        return self.parent_item


class HDF5TreeModel(QAbstractItemModel):
    """A model to represent the hierarchical structure of an HDF5 file."""

    def __init__(self, hdf5_file, parent=None):
        super(HDF5TreeModel, self).__init__(parent)
        self.root_item = HDF5TreeItem("HDF5 File")
        self.setup_model_data(hdf5_file, self.root_item)

    def setup_model_data(self, obj, parent):
        """Recursively populate the model with HDF5 groups and datasets."""
        if isinstance(obj, h5py.Group):
            for key, item in obj.items():
                child_item = HDF5TreeItem(name=key, obj=item, parent=parent)
                parent.append_child(child_item)
                self.setup_model_data(item, child_item)
        elif isinstance(obj, h5py.Dataset):
            parent.append_child(HDF5TreeItem(name=obj.name, obj=obj, parent=parent))

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            return self.root_item.child_count()
        parent_item = parent.internalPointer()
        return parent_item.child_count()

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.DisplayRole:
            return None
        item = index.internalPointer()
        return item.data()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self.root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)


class HDF5_Browser(QWidget):
    def __init__(self, brain_viewer, workdir=None):
        super().__init__()
        self.nbv = brain_viewer

        # Title and Layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("HDF5 Browser"))

        # file selection
        if workdir is None:
            workdir = Path.home()
        path, _ = QFileDialog.getOpenFileUrl(
            self,
            caption="Select .h5 file",
            directory=QUrl(str(workdir)),
            # filter = "HDF5 files (*.h5, *.hdf5)",
        )
        self.file_path = Path(path.path())
        self.layout().addWidget(QLabel(str(self.file_path)))

        # display hierchical file content
        self.file = h5py.File(self.file_path, "r")
        self.model = HDF5TreeModel(self.file)
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.layout().addWidget(self.tree_view)
        self.tree_view.setSelectionMode(QAbstractItemView.SingleSelection)

        button = QPushButton("Load as new Layer")
        button.clicked.connect(self._load_data)
        self.layout().addWidget(button)

    def _load_data(self):
        obj = self._get_selected_dataset()
        if not isinstance(obj, h5py.Dataset):
            return None

        ndim = obj.ndim
        if ndim > 2:
            action = "image"
        else:
            if obj.shape[1] <= 3:
                action = "scatter"
            else:
                action = "image"

        if action == "scatter":
            show_info("Loading points to scatter.")
            self.nbv.points(obj[()], size=4)

        elif action == "image":
            show_info("Loading image.")

    def _get_selected_dataset(self):
        selected = self.tree_view.selectedIndexes()
        assert len(selected) == 1
        selected = selected[0]
        item = selected.internalPointer()
        if not item.type == h5py.Dataset:
            show_error("Can only select datasets")
        obj = item.obj
        return obj
