"""FSC + Local Resolution — ChimeraX GUI tool."""

import numpy as np
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QDoubleSpinBox, QLabel, QSizePolicy,
    QRadioButton, QButtonGroup, QTabWidget, QWidget, QProgressBar,
    QApplication, QCheckBox,
)
from Qt.QtCore import Qt


class FSCTool(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)

        # FSC state
        self._fsc_values = None
        self._pixel_size = None
        self._sim_map = None

        # Local resolution state
        self._lr_output = None
        self._lr_vol = None   # the ChimeraX volume created from lr output

        # Directional resolution state
        self._dr_output = None   # dict: metric key -> per-voxel array
        self._dr_vol = None

        self._build_ui()
        self.tool_window.manage('side')
        self._refresh_volume_list()
        self._refresh_structure_list()
        self._refresh_lr_volume_list()
        self._refresh_dr_volume_list()
        self._handlers = [
            session.triggers.add_handler('add models', self._on_models_changed),
            session.triggers.add_handler('remove models', self._on_models_changed),
        ]

    # ==================================================================
    # Top-level UI — two tabs
    # ==================================================================

    def _build_ui(self):
        parent = self.tool_window.ui_area
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()

        fsc_widget = QWidget()
        fsc_layout = QVBoxLayout()
        fsc_layout.setContentsMargins(6, 6, 6, 6)
        fsc_layout.setSpacing(4)
        self._build_fsc_tab(fsc_layout)
        fsc_widget.setLayout(fsc_layout)
        tabs.addTab(fsc_widget, "Global FSC")

        lr_widget = QWidget()
        lr_layout = QVBoxLayout()
        lr_layout.setContentsMargins(6, 6, 6, 6)
        lr_layout.setSpacing(4)
        self._build_localres_tab(lr_layout)
        lr_widget.setLayout(lr_layout)
        tabs.addTab(lr_widget, "Local Resolution")

        dr_widget = QWidget()
        dr_layout = QVBoxLayout()
        dr_layout.setContentsMargins(6, 6, 6, 6)
        dr_layout.setSpacing(4)
        self._build_directional_tab(dr_layout)
        dr_widget.setLayout(dr_layout)
        tabs.addTab(dr_widget, "Directional Resolution")

        root.addWidget(tabs)
        parent.setLayout(root)

    # ==================================================================
    # Tab 1 — Global FSC
    # ==================================================================

    def _build_fsc_tab(self, layout):
        # Mode toggle
        mode_row = QHBoxLayout()
        self._mode_map = QRadioButton("Map vs Map")
        self._mode_pdb = QRadioButton("Map vs Atomic model")
        self._mode_map.setChecked(True)
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self._mode_map)
        self._mode_group.addButton(self._mode_pdb)
        mode_row.addWidget(QLabel("Mode:"))
        mode_row.addWidget(self._mode_map)
        mode_row.addWidget(self._mode_pdb)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        form = QFormLayout()
        form.setSpacing(4)

        self._map1_combo = QComboBox()
        self._map2_combo = QComboBox()
        self._mask_combo = QComboBox()
        self._struct_combo = QComboBox()
        # Keep all dropdowns the same width regardless of item text length
        for combo in (self._map1_combo, self._map2_combo,
                      self._mask_combo, self._struct_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._map1_combo.currentIndexChanged.connect(self._auto_fill_pixel_size)
        self._map2_combo.currentIndexChanged.connect(self._auto_fill_pixel_size)

        self._pix_spin = QDoubleSpinBox()
        self._pix_spin.setRange(0.001, 100.0)
        self._pix_spin.setDecimals(4)
        self._pix_spin.setSingleStep(0.001)
        self._pix_spin.setValue(1.0)
        self._pix_spin.setSuffix("  Å/vox")

        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.001, 1.0)
        self._thresh_spin.setDecimals(3)
        self._thresh_spin.setSingleStep(0.001)
        self._thresh_spin.setValue(0.143)
        self._thresh_spin.valueChanged.connect(self._on_threshold_changed)

        self._sim_res_spin = QDoubleSpinBox()
        self._sim_res_spin.setRange(0.5, 50.0)
        self._sim_res_spin.setDecimals(3)
        self._sim_res_spin.setSingleStep(0.1)
        self._sim_res_spin.setValue(2.0)
        self._sim_res_spin.setSuffix("  Å")

        self._map2_label = QLabel("Map 2:")
        self._struct_label = QLabel("Atomic model:")
        self._sim_res_label = QLabel("Sim. resolution:")

        form.addRow("Map 1:", self._map1_combo)
        form.addRow(self._map2_label, self._map2_combo)
        form.addRow(self._struct_label, self._struct_combo)
        form.addRow(self._sim_res_label, self._sim_res_spin)
        form.addRow("Mask (opt.):", self._mask_combo)
        form.addRow("Pixel size:", self._pix_spin)
        form.addRow("Threshold:", self._thresh_spin)
        layout.addLayout(form)

        self._compute_btn = QPushButton("Compute FSC")
        self._compute_btn.clicked.connect(self._on_compute)
        layout.addWidget(self._compute_btn)

        save_row = QHBoxLayout()
        self._save_img_btn = QPushButton("Save image")
        self._save_data_btn = QPushButton("Save data (CSV)")
        self._save_img_btn.clicked.connect(self._on_save_image)
        self._save_data_btn.clicked.connect(self._on_save_data)
        save_row.addWidget(self._save_img_btn)
        save_row.addWidget(self._save_data_btn)
        layout.addLayout(save_row)

        self._res_label = QLabel("Resolution at threshold:  —")
        self._res_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._res_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Computing Global FSC…  %p%")
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        layout.addWidget(self._make_fsc_canvas())

        self._mode_map.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()

    def _on_mode_changed(self):
        map_mode = self._mode_map.isChecked()
        self._map2_label.setVisible(map_mode)
        self._map2_combo.setVisible(map_mode)
        self._struct_label.setVisible(not map_mode)
        self._struct_combo.setVisible(not map_mode)
        self._sim_res_label.setVisible(not map_mode)
        self._sim_res_spin.setVisible(not map_mode)
        self._thresh_spin.setValue(0.143 if map_mode else 0.5)

    def _make_fsc_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        self._fig = Figure(figsize=(5, 3.5), constrained_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._draw_empty_fsc_axes()
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return self._canvas

    def _draw_empty_fsc_axes(self):
        ax = self._ax
        ax.cla()
        ax.set_xlabel("Spatial frequency (1/Å)", fontsize=9)
        ax.set_ylabel("FSC", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.143, color='tomato', linestyle='--', linewidth=1,
                   label='0.143 threshold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    # ==================================================================
    # Tab 2 — Local Resolution
    # ==================================================================

    def _build_localres_tab(self, layout):
        hint = QLabel("Two half maps (recommended) or one map. A mask is required.")
        hint.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(4)

        self._lr_map1_combo = QComboBox()
        self._lr_map2_combo = QComboBox()
        self._lr_mask_combo = QComboBox()
        self._lr_map1_combo.currentIndexChanged.connect(self._lr_auto_fill_pixel_size)

        self._lr_pix_spin = QDoubleSpinBox()
        self._lr_pix_spin.setRange(0.001, 100.0)
        self._lr_pix_spin.setDecimals(4)
        self._lr_pix_spin.setSingleStep(0.001)
        self._lr_pix_spin.setValue(1.0)
        self._lr_pix_spin.setSuffix("  Å/vox")

        self._lr_maxres_spin = QDoubleSpinBox()
        self._lr_maxres_spin.setRange(0.1, 200.0)
        self._lr_maxres_spin.setDecimals(2)
        self._lr_maxres_spin.setValue(15.0)
        self._lr_maxres_spin.setSuffix(" Å")

        self._lr_step_spin = QDoubleSpinBox()
        self._lr_step_spin.setRange(0.05, 10.0)
        self._lr_step_spin.setDecimals(2)
        self._lr_step_spin.setValue(0.5)
        self._lr_step_spin.setSuffix(" Å")

        self._lr_signif_spin = QDoubleSpinBox()
        self._lr_signif_spin.setRange(0.5, 0.999)
        self._lr_signif_spin.setDecimals(3)
        self._lr_signif_spin.setValue(0.95)

        form.addRow("Map 1:", self._lr_map1_combo)
        form.addRow("Map 2 (opt.):", self._lr_map2_combo)
        form.addRow("Mask:", self._lr_mask_combo)
        form.addRow("Pixel size:", self._lr_pix_spin)
        form.addRow("Max resolution:", self._lr_maxres_spin)
        form.addRow("Step size:", self._lr_step_spin)
        form.addRow("Significance:", self._lr_signif_spin)
        layout.addLayout(form)

        self._lr_compute_btn = QPushButton("Compute Local Resolution")
        self._lr_compute_btn.clicked.connect(self._on_lr_compute)
        layout.addWidget(self._lr_compute_btn)

        self._lr_progress = QProgressBar()
        self._lr_progress.setRange(0, 100)
        self._lr_progress.setTextVisible(True)
        self._lr_progress.setFormat("Computing Local Resolution…  %p%")
        self._lr_progress.setVisible(False)
        layout.addWidget(self._lr_progress)

        self._lr_status_label = QLabel("")
        self._lr_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lr_status_label)

        # Resolution range controls
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Display range:"))
        self._lr_res_min_spin = QDoubleSpinBox()
        self._lr_res_min_spin.setRange(0.1, 100.0)
        self._lr_res_min_spin.setDecimals(2)
        self._lr_res_min_spin.setValue(2.0)
        self._lr_res_min_spin.setSuffix(" Å")
        self._lr_res_max_spin = QDoubleSpinBox()
        self._lr_res_max_spin.setRange(0.1, 100.0)
        self._lr_res_max_spin.setDecimals(2)
        self._lr_res_max_spin.setValue(12.0)
        self._lr_res_max_spin.setSuffix(" Å")
        range_row.addWidget(self._lr_res_min_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._lr_res_max_spin)
        range_row.addStretch()
        layout.addLayout(range_row)

        self._lr_update_btn = QPushButton("Update Histogram")
        self._lr_update_btn.clicked.connect(self._on_lr_update_histogram)
        self._lr_update_btn.setEnabled(False)
        layout.addWidget(self._lr_update_btn)

        layout.addWidget(self._make_lr_canvas())

        self._lr_color_btn = QPushButton("Color Map in ChimeraX")
        self._lr_color_btn.clicked.connect(self._on_lr_color_map)
        self._lr_color_btn.setEnabled(False)
        layout.addWidget(self._lr_color_btn)

    def _make_lr_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.gridspec import GridSpec

        self._lr_fig = Figure(figsize=(5, 3.5), constrained_layout=True)
        gs = GridSpec(2, 1, figure=self._lr_fig,
                      height_ratios=[12, 1], hspace=0.05)
        self._lr_ax_hist = self._lr_fig.add_subplot(gs[0])
        self._lr_ax_cbar = self._lr_fig.add_subplot(gs[1])

        self._lr_canvas = FigureCanvas(self._lr_fig)
        self._lr_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draw_empty_lr_axes()
        return self._lr_canvas

    def _draw_empty_lr_axes(self):
        self._lr_ax_hist.cla()
        self._lr_ax_hist.set_xlabel("Local resolution (Å)", fontsize=9)
        self._lr_ax_hist.set_ylabel("Voxel count", fontsize=9)
        self._lr_ax_hist.text(0.5, 0.5, "No data yet",
                              transform=self._lr_ax_hist.transAxes,
                              ha='center', va='center',
                              fontsize=10, color='grey')
        self._lr_ax_hist.grid(True, alpha=0.25)
        self._lr_ax_cbar.set_visible(False)
        self._lr_canvas.draw()

    # ------------------------------------------------------------------
    # Local resolution — volume list
    # ------------------------------------------------------------------

    def _refresh_lr_volume_list(self):
        from chimerax.map import Volume
        vols = self.session.models.list(type=Volume)

        for combo in (self._lr_map1_combo, self._lr_map2_combo):
            prev = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if combo is self._lr_map2_combo:
                combo.addItem("None", userData=None)
            sel = 0
            for idx, v in enumerate(vols):
                combo.addItem(f"#{v.id_string}  {v.name}", userData=v)
                if v is prev:
                    sel = idx + (1 if combo is self._lr_map2_combo else 0)
            combo.setCurrentIndex(sel)
            combo.blockSignals(False)

        prev_mask = self._lr_mask_combo.currentData()
        self._lr_mask_combo.blockSignals(True)
        self._lr_mask_combo.clear()
        self._lr_mask_combo.addItem("None", userData=None)
        sel = 0
        for idx, v in enumerate(vols):
            self._lr_mask_combo.addItem(f"#{v.id_string}  {v.name}", userData=v)
            if v is prev_mask:
                sel = idx + 1
        self._lr_mask_combo.setCurrentIndex(sel)
        self._lr_mask_combo.blockSignals(False)

        self._lr_auto_fill_pixel_size()

    def _lr_auto_fill_pixel_size(self):
        vol = self._lr_map1_combo.currentData()
        if vol is not None:
            try:
                step = float(vol.data.step[0])
                if step > 0:
                    self._lr_pix_spin.setValue(step)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Local resolution — compute
    # ------------------------------------------------------------------

    def _on_lr_compute(self):
        from .localres_core import monores

        map1 = self._lr_map1_combo.currentData()
        map2 = self._lr_map2_combo.currentData()
        mask = self._lr_mask_combo.currentData()

        if map1 is None:
            self.session.logger.error("Local Resolution: please select Map 1.")
            return
        # MonoRes always needs a mask (defines the protein / noise regions)
        if mask is None:
            msg = "A mask is required for MonoRes."
            self.session.logger.error(f"Local Resolution: {msg}")
            self._lr_status_label.setText(msg)
            self._lr_status_label.setStyleSheet("color: #c0392b;")
            return

        self._lr_compute_btn.setEnabled(False)
        self._lr_status_label.setText("Computing…")
        self._lr_status_label.setStyleSheet("")
        self._lr_progress.setValue(0)
        self._lr_progress.setVisible(True)
        self.session.logger.status("Computing Local Resolution …")

        # Read map data on the main thread, then compute on a worker thread so
        # ChimeraX stays responsive. UI updates are marshalled back with
        # session.ui.thread_safe (Qt widgets may only be touched on the GUI thread).
        vol1 = map1.data.matrix().astype(np.float64, copy=True)
        vol2 = map2.data.matrix().astype(np.float64, copy=True) if map2 is not None else None
        mask_arr = mask.data.matrix().astype(np.float64, copy=True)
        pixel_size = self._lr_pix_spin.value()
        max_res = self._lr_maxres_spin.value()
        step = self._lr_step_spin.value()
        significance = self._lr_signif_spin.value()

        import threading, time

        def progress(frac):
            pct = int(max(0.0, min(1.0, frac)) * 100)
            self.session.ui.thread_safe(self._lr_set_progress, pct)

        def worker():
            t0 = time.perf_counter()
            try:
                output = monores(
                    vol1, vol2, mask_arr,
                    sampling_rate=pixel_size,
                    max_res=max_res, step=step, significance=significance,
                    progress_callback=progress,
                )
                elapsed = time.perf_counter() - t0
                self.session.ui.thread_safe(self._lr_on_done, output, elapsed)
            except Exception as e:
                self.session.ui.thread_safe(self._lr_on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _lr_set_progress(self, pct):
        self._lr_progress.setValue(pct)
        self._lr_progress.setFormat("Computing Local Resolution…  %p%")

    def _lr_on_done(self, output, elapsed):
        self._lr_output = output
        valid = output[np.isfinite(output)]
        data_min = float(np.percentile(valid, 2))
        data_max = float(np.percentile(valid, 98))
        self._lr_res_min_spin.setValue(round(data_min, 2))
        self._lr_res_max_spin.setValue(round(data_max, 2))

        self._update_lr_histogram(output, data_min, data_max)

        self._lr_update_btn.setEnabled(True)
        self._lr_color_btn.setEnabled(True)
        self._lr_compute_btn.setEnabled(True)
        self._lr_progress.setVisible(False)
        self._lr_status_label.setText(
            f"Done in {elapsed:.1f} s — resolution range "
            f"{data_min:.1f}–{data_max:.1f} Å"
        )
        self._lr_status_label.setStyleSheet("color: #27ae60;")
        self.session.logger.info(
            f"Local resolution computed in {elapsed:.1f} s.\n"
            f"  Output shape: {output.shape}\n"
            f"  Range (2nd–98th pct): {data_min:.2f} – {data_max:.2f} Å"
        )
        self.session.logger.status("Local resolution done.")

    def _lr_on_error(self, msg):
        self.session.logger.error(f"Local resolution failed: {msg}")
        self._lr_status_label.setText(f"Error: {msg}")
        self._lr_status_label.setStyleSheet("color: #c0392b;")
        self._lr_compute_btn.setEnabled(True)
        self._lr_progress.setVisible(False)

    def _on_lr_update_histogram(self):
        if self._lr_output is None:
            return
        self._update_lr_histogram(
            self._lr_output,
            self._lr_res_min_spin.value(),
            self._lr_res_max_spin.value(),
        )

    def _update_lr_histogram(self, output, res_min, res_max):
        self._draw_metric_histogram(self._lr_ax_hist, self._lr_ax_cbar,
                                    self._lr_canvas, output, res_min, res_max,
                                    "Local resolution (Å)")

    def _draw_metric_histogram(self, ax, cax, canvas, output, vmin, vmax, xlabel):
        """Shared histogram + matching colorbar used by the Local and
        Directional tabs. Bars are colored by value with matplotlib 'rainbow'."""
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors

        if vmax <= vmin:
            vmax = vmin + 1e-6

        flat = output.ravel()
        flat = flat[np.isfinite(flat)]
        in_range = flat[(flat >= vmin) & (flat <= vmax)]

        cmap = mcm.get_cmap('rainbow')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        ax.cla()
        if len(in_range) == 0:
            ax.text(0.5, 0.5, "No voxels in selected range",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='grey')
        else:
            counts, edges = np.histogram(in_range, bins=60, range=(vmin, vmax))
            centers = (edges[:-1] + edges[1:]) / 2
            bar_colors = [cmap(norm(c)) for c in centers]
            ax.bar(edges[:-1], counts, width=np.diff(edges),
                   color=bar_colors, align='edge', edgecolor='none')

        ax.set_ylabel("Voxel count", fontsize=9)
        ax.set_xlim(vmin, vmax)
        ax.set_xticklabels([])  # x labels live on the colorbar below
        ax.grid(True, axis='y', alpha=0.2)

        # Colorbar directly beneath, sharing the same x-range → reads as one unit
        cax.set_visible(True)
        cax.cla()
        gradient = np.linspace(vmin, vmax, 256).reshape(1, -1)
        cax.imshow(gradient, aspect='auto', cmap=cmap, extent=[vmin, vmax, 0, 1])
        cax.set_yticks([])
        cax.set_xlim(vmin, vmax)
        cax.set_xlabel(xlabel, fontsize=9)

        canvas.draw()

    # ------------------------------------------------------------------
    # Local resolution — color map in ChimeraX
    # ------------------------------------------------------------------

    def _on_lr_color_map(self):
        if self._lr_output is None:
            self.session.logger.error("Run Local Resolution first.")
            return
        map1 = self._lr_map1_combo.currentData()
        if map1 is None:
            return
        res_min = self._lr_res_min_spin.value()
        res_max = self._lr_res_max_spin.value()
        result = self._color_map_by_array(
            map1, self._lr_output, res_min, res_max,
            values_name="resolution values", copy_name="local resolution",
            prev_attr="_lr_vol", key_title="Local resolution (Å)")
        if result is not None:
            map_copy, vals_vol = result
            self.session.logger.info(
                f"Colored map #{map_copy.id_string} ({map1.name}) "
                f"by local resolution, range {res_min:.1f}–{res_max:.1f} Å.\n"
                f"Resolution values volume: #{vals_vol.id_string} (hidden)"
            )

    def _color_map_by_array(self, map1, array, vmin, vmax,
                            values_name, copy_name, prev_attr, key_title):
        """Shared by the Local and Directional tabs: build a ChimeraX volume
        from a per-voxel array, copy `map1`, and color the copy's surface by
        the array using a palette that matches the matplotlib histogram.

        `prev_attr` names the attribute holding the previously created values
        volume for this tab so repeated clicks replace rather than stack.
        Returns (map_copy, values_volume) or None on failure.
        """
        from chimerax.map_data import ArrayGridData
        from chimerax.map import volume_from_grid_data, Volume
        from chimerax.core.commands import run

        # Replace (not stack) the values volume and colored copy from a previous click
        for attr in (prev_attr, prev_attr + "_copy"):
            prev = getattr(self, attr, None)
            if prev is not None:
                try:
                    if not prev.deleted:
                        run(self.session, f"close #{prev.id_string}")
                except Exception:
                    pass
                setattr(self, attr, None)

        # NaN (outside mask) -> vmax so those voxels don't skew the coloring.
        arr = np.array(array, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, vmax)
        clipped = np.clip(arr, vmin, vmax).astype(np.float32)
        grid = ArrayGridData(clipped, origin=map1.data.origin,
                             step=map1.data.step, name=values_name)
        vals_vol = volume_from_grid_data(grid, self.session, open_model=True)
        setattr(self, prev_attr, vals_vol)

        # Copy map1 so the original stays untouched. Build the copy ourselves
        # with its final name: Volume Viewer reads the name only once, when the
        # model is added, and does not pick up later renames.
        copy_grid = ArrayGridData(map1.data.matrix().copy(),
                                  origin=map1.data.origin,
                                  step=map1.data.step, name=copy_name)
        map_copy = volume_from_grid_data(copy_grid, self.session, open_model=True)
        if map_copy is None:
            self.session.logger.error("Could not create a copy of the map.")
            return None
        map_copy.name = copy_name
        setattr(self, prev_attr + "_copy", map_copy)

        # The copy needs a surface with real geometry before 'color sample'
        # can paint it. Use the original's level if it has one, else statistics.
        level = None
        try:
            surfs = map1.surfaces
            if surfs:
                level = float(surfs[0].level)
        except Exception:
            level = None
        if level is None:
            data = map_copy.data.matrix()
            level = float(np.nanmean(data) + 3.0 * np.nanstd(data))
        run(self.session, f"volume #{map_copy.id_string} level {level} style surface")

        # ChimeraX computes big surfaces in a background thread; force them to
        # finish synchronously so the geometry exists when we color it.
        try:
            ro = map_copy.rendering_options
            for s in map_copy.surfaces:
                s.set_level(s.level, use_thread=False)
                s.update_surface(ro)
        except Exception as e:
            self.session.logger.warning(f"Surface pre-compute skipped: {e}")

        # Palette sampled from matplotlib 'rainbow' so the map matches the histogram
        palette = self._matplotlib_palette('rainbow', vmin, vmax, n=32)
        run(self.session,
            f"color sample #{map_copy.id_string} map #{vals_vol.id_string} "
            f"palette {palette} range {vmin},{vmax}")

        # Hide the raw values volume so it doesn't obscure the colored map
        run(self.session, f"hide #{vals_vol.id_string} models")

        self._show_color_key(vmin, vmax, key_title)
        return map_copy, vals_vol

    def _show_color_key(self, vmin, vmax, title, n=32, n_labels=5):
        """Draw the same matplotlib 'rainbow' colorbar in the 3D view (bottom
        left) so the color scale is visible while rotating the structure."""
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        from chimerax.core.commands import run

        cmap = mcm.get_cmap('rainbow')
        label_idx = {round(i * (n - 1) / (n_labels - 1)) for i in range(n_labels)}
        stops = []
        for i in range(n):
            frac = i / (n - 1)
            hex_color = mcolors.to_hex(cmap(frac))
            label = f"{vmin + frac * (vmax - vmin):.2f}" if i in label_idx else ""
            stops.append(f"{hex_color}:{label}")
        run(self.session,
            "key " + " ".join(stops) +
            " pos 0.05,0.05 size 0.4,0.04 fontSize 14 colorTreatment blended")

        # Title above the key: create once, then just change its text
        try:
            run(self.session, f'2dlabels change fsc_key_title text "{title}"')
        except Exception:
            run(self.session, f'2dlabels create fsc_key_title text "{title}" '
                              f'xpos 0.05 ypos 0.12 size 14')

    # ==================================================================
    # Tab 3 — Directional Resolution (MonoDir)
    # ==================================================================

    def _build_directional_tab(self, layout):
        from .directional_core import METRICS

        hint = QLabel("One map + a mask. MonoRes runs first internally, then MonoDir.")
        hint.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(4)

        self._dr_map_combo = QComboBox()
        self._dr_mask_combo = QComboBox()
        for combo in (self._dr_map_combo, self._dr_mask_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._dr_map_combo.currentIndexChanged.connect(self._dr_auto_fill_pixel_size)

        self._dr_pix_spin = QDoubleSpinBox()
        self._dr_pix_spin.setRange(0.001, 100.0)
        self._dr_pix_spin.setDecimals(4)
        self._dr_pix_spin.setSingleStep(0.001)
        self._dr_pix_spin.setValue(1.0)
        self._dr_pix_spin.setSuffix("  Å/vox")

        self._dr_signif_spin = QDoubleSpinBox()
        self._dr_signif_spin.setRange(0.5, 0.999)
        self._dr_signif_spin.setDecimals(3)
        self._dr_signif_spin.setValue(0.95)

        self._dr_step_spin = QDoubleSpinBox()
        self._dr_step_spin.setRange(0.05, 10.0)
        self._dr_step_spin.setDecimals(2)
        self._dr_step_spin.setValue(0.5)
        self._dr_step_spin.setSuffix(" Å")

        self._dr_fast_check = QCheckBox("Fast computation (recommended for large volumes)")
        self._dr_premask_check = QCheckBox("Input map is premasked (spherical mask)")
        self._dr_premask_check.toggled.connect(self._on_dr_premask_toggled)

        self._dr_radius_label = QLabel("Spherical mask radius:")
        self._dr_radius_spin = QDoubleSpinBox()
        self._dr_radius_spin.setRange(-1.0, 100000.0)
        self._dr_radius_spin.setDecimals(1)
        self._dr_radius_spin.setValue(-1.0)
        self._dr_radius_spin.setSuffix(" px")
        self._dr_radius_spin.setToolTip("-1 = use half the box size")

        form.addRow("Map:", self._dr_map_combo)
        form.addRow("Mask:", self._dr_mask_combo)
        form.addRow("Pixel size:", self._dr_pix_spin)
        form.addRow("Significance:", self._dr_signif_spin)
        form.addRow("Resolution step:", self._dr_step_spin)
        form.addRow("", self._dr_fast_check)
        form.addRow("", self._dr_premask_check)
        form.addRow(self._dr_radius_label, self._dr_radius_spin)
        layout.addLayout(form)

        self._dr_compute_btn = QPushButton("Compute Directional Resolution")
        self._dr_compute_btn.clicked.connect(self._on_dr_compute)
        layout.addWidget(self._dr_compute_btn)

        self._dr_progress = QProgressBar()
        self._dr_progress.setRange(0, 100)
        self._dr_progress.setTextVisible(True)
        self._dr_progress.setFormat("Computing Directional Resolution…  %p%")
        self._dr_progress.setVisible(False)
        layout.addWidget(self._dr_progress)

        self._dr_status_label = QLabel("")
        self._dr_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._dr_status_label)

        # Metric selector — which per-voxel output to histogram / color by
        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Metric:"))
        self._dr_metric_combo = QComboBox()
        for key, (label, _units, _rng) in METRICS.items():
            self._dr_metric_combo.addItem(label, userData=key)
        self._dr_metric_combo.currentIndexChanged.connect(self._on_dr_metric_changed)
        metric_row.addWidget(self._dr_metric_combo, 1)
        layout.addLayout(metric_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Display range:"))
        self._dr_min_spin = QDoubleSpinBox()
        self._dr_min_spin.setRange(0.0, 1000.0)
        self._dr_min_spin.setDecimals(2)
        self._dr_max_spin = QDoubleSpinBox()
        self._dr_max_spin.setRange(0.0, 1000.0)
        self._dr_max_spin.setDecimals(2)
        self._dr_max_spin.setValue(1.0)
        range_row.addWidget(self._dr_min_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._dr_max_spin)
        range_row.addStretch()
        layout.addLayout(range_row)

        self._dr_update_btn = QPushButton("Update Histogram")
        self._dr_update_btn.clicked.connect(self._on_dr_update_histogram)
        self._dr_update_btn.setEnabled(False)
        layout.addWidget(self._dr_update_btn)

        layout.addWidget(self._make_dr_canvas())

        self._dr_color_btn = QPushButton("Color Map in ChimeraX")
        self._dr_color_btn.clicked.connect(self._on_dr_color_map)
        self._dr_color_btn.setEnabled(False)
        layout.addWidget(self._dr_color_btn)

        self._on_dr_premask_toggled(False)

    def _on_dr_premask_toggled(self, checked):
        self._dr_radius_label.setVisible(bool(checked))
        self._dr_radius_spin.setVisible(bool(checked))

    def _make_dr_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.gridspec import GridSpec

        self._dr_fig = Figure(figsize=(5, 3.5), constrained_layout=True)
        gs = GridSpec(2, 1, figure=self._dr_fig, height_ratios=[12, 1], hspace=0.05)
        self._dr_ax_hist = self._dr_fig.add_subplot(gs[0])
        self._dr_ax_cbar = self._dr_fig.add_subplot(gs[1])

        self._dr_canvas = FigureCanvas(self._dr_fig)
        self._dr_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draw_empty_dr_axes()
        return self._dr_canvas

    def _draw_empty_dr_axes(self):
        self._dr_ax_hist.cla()
        self._dr_ax_hist.set_xlabel("Metric value", fontsize=9)
        self._dr_ax_hist.set_ylabel("Voxel count", fontsize=9)
        self._dr_ax_hist.text(0.5, 0.5, "No data yet",
                              transform=self._dr_ax_hist.transAxes,
                              ha='center', va='center', fontsize=10, color='grey')
        self._dr_ax_hist.grid(True, alpha=0.25)
        self._dr_ax_cbar.set_visible(False)
        self._dr_canvas.draw()

    def _refresh_dr_volume_list(self):
        from chimerax.map import Volume
        vols = self.session.models.list(type=Volume)

        prev = self._dr_map_combo.currentData()
        self._dr_map_combo.blockSignals(True)
        self._dr_map_combo.clear()
        sel = 0
        for idx, v in enumerate(vols):
            self._dr_map_combo.addItem(f"#{v.id_string}  {v.name}", userData=v)
            if v is prev:
                sel = idx
        self._dr_map_combo.setCurrentIndex(sel)
        self._dr_map_combo.blockSignals(False)

        prev_mask = self._dr_mask_combo.currentData()
        self._dr_mask_combo.blockSignals(True)
        self._dr_mask_combo.clear()
        self._dr_mask_combo.addItem("None", userData=None)
        sel = 0
        for idx, v in enumerate(vols):
            self._dr_mask_combo.addItem(f"#{v.id_string}  {v.name}", userData=v)
            if v is prev_mask:
                sel = idx + 1
        self._dr_mask_combo.setCurrentIndex(sel)
        self._dr_mask_combo.blockSignals(False)

        self._dr_auto_fill_pixel_size()

    def _dr_auto_fill_pixel_size(self):
        vol = self._dr_map_combo.currentData()
        if vol is not None:
            try:
                step = float(vol.data.step[0])
                if step > 0:
                    self._dr_pix_spin.setValue(step)
            except Exception:
                pass

    # ---- compute ------------------------------------------------------

    def _on_dr_compute(self):
        from .directional_core import fake_monodir

        map1 = self._dr_map_combo.currentData()
        mask = self._dr_mask_combo.currentData()
        if map1 is None:
            self.session.logger.error("Directional Resolution: please select a map.")
            return
        if mask is None:
            msg = "A mask is required for MonoDir."
            self.session.logger.error(f"Directional Resolution: {msg}")
            self._dr_status_label.setText(msg)
            self._dr_status_label.setStyleSheet("color: #c0392b;")
            return

        self._dr_compute_btn.setEnabled(False)
        self._dr_status_label.setText("Computing…")
        self._dr_status_label.setStyleSheet("")
        self._dr_progress.setValue(0)
        self._dr_progress.setVisible(True)
        self.session.logger.status("Computing Directional Resolution …")

        vol = map1.data.matrix().astype(np.float64, copy=True)
        mask_arr = mask.data.matrix().astype(np.float64, copy=True)
        pixel_size = self._dr_pix_spin.value()
        significance = self._dr_signif_spin.value()
        res_step = self._dr_step_spin.value()
        fast = self._dr_fast_check.isChecked()
        premasked = self._dr_premask_check.isChecked()
        volume_radius = self._dr_radius_spin.value()

        import threading, time

        def progress(frac):
            pct = int(max(0.0, min(1.0, frac)) * 100)
            self.session.ui.thread_safe(self._dr_set_progress, pct)

        def worker():
            t0 = time.perf_counter()
            try:
                out = fake_monodir(
                    vol, mask_arr, sampling_rate=pixel_size,
                    significance=significance, res_step=res_step, fast=fast,
                    premasked=premasked, volume_radius=volume_radius,
                    progress_callback=progress,
                )
                elapsed = time.perf_counter() - t0
                self.session.ui.thread_safe(self._dr_on_done, out, elapsed)
            except Exception as e:
                self.session.ui.thread_safe(self._dr_on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _dr_set_progress(self, pct):
        self._dr_progress.setValue(pct)
        self._dr_progress.setFormat("Computing Directional Resolution…  %p%")

    def _dr_on_done(self, out, elapsed):
        self._dr_output = out
        self._dr_apply_metric_range()
        self._dr_update_btn.setEnabled(True)
        self._dr_color_btn.setEnabled(True)
        self._dr_compute_btn.setEnabled(True)
        self._dr_progress.setVisible(False)
        self._dr_status_label.setText(f"Done in {elapsed:.1f} s")
        self._dr_status_label.setStyleSheet("color: #27ae60;")
        from .directional_core import METRICS
        lines = [f"Directional resolution computed in {elapsed:.1f} s."]
        for key, arr in out.items():
            v = arr[np.isfinite(arr)]
            if v.size:
                lines.append(f"  {METRICS[key][0]:<32} "
                             f"{float(np.percentile(v, 2)):.2f} – "
                             f"{float(np.percentile(v, 98)):.2f}")
        self.session.logger.info("\n".join(lines))
        self.session.logger.status("Directional resolution done.")

    def _dr_on_error(self, msg):
        self.session.logger.error(f"Directional resolution failed: {msg}")
        self._dr_status_label.setText(f"Error: {msg}")
        self._dr_status_label.setStyleSheet("color: #c0392b;")
        self._dr_compute_btn.setEnabled(True)
        self._dr_progress.setVisible(False)

    # ---- metric / histogram / color ----------------------------------

    def _dr_current_key(self):
        return self._dr_metric_combo.currentData()

    def _dr_current_array(self):
        if self._dr_output is None:
            return None
        return self._dr_output.get(self._dr_current_key())

    def _dr_apply_metric_range(self):
        """Set the display range to suit the selected metric, then redraw."""
        from .directional_core import METRICS
        arr = self._dr_current_array()
        if arr is None:
            return
        fixed = METRICS[self._dr_current_key()][2]
        if fixed is not None:
            vmin, vmax = fixed
        else:
            v = arr[np.isfinite(arr)]
            vmin = float(np.percentile(v, 2)) if v.size else 0.0
            vmax = float(np.percentile(v, 98)) if v.size else 1.0
        for spin, val in ((self._dr_min_spin, vmin), (self._dr_max_spin, vmax)):
            spin.blockSignals(True)
            spin.setValue(round(val, 2))
            spin.blockSignals(False)
        self._dr_draw_histogram(vmin, vmax)

    def _dr_draw_histogram(self, vmin, vmax):
        from .directional_core import METRICS
        arr = self._dr_current_array()
        if arr is None:
            return
        xlabel = METRICS[self._dr_current_key()][1]
        self._draw_metric_histogram(self._dr_ax_hist, self._dr_ax_cbar,
                                    self._dr_canvas, arr, vmin, vmax, xlabel)

    def _on_dr_metric_changed(self, *args):
        if self._dr_output is not None:
            self._dr_apply_metric_range()

    def _on_dr_update_histogram(self):
        self._dr_draw_histogram(self._dr_min_spin.value(), self._dr_max_spin.value())

    def _on_dr_color_map(self):
        from .directional_core import METRICS
        arr = self._dr_current_array()
        if arr is None:
            self.session.logger.error("Run Directional Resolution first.")
            return
        map1 = self._dr_map_combo.currentData()
        if map1 is None:
            return
        key = self._dr_current_key()
        label, units, _ = METRICS[key]
        vmin = self._dr_min_spin.value()
        vmax = self._dr_max_spin.value()
        result = self._color_map_by_array(
            map1, arr, vmin, vmax,
            values_name=f"{label} values", copy_name=f"directional: {label}",
            prev_attr="_dr_vol",
            key_title=label if key == "doa" else f"{label} (Å)")
        if result is not None:
            map_copy, vals_vol = result
            self.session.logger.info(
                f"Colored map #{map_copy.id_string} ({map1.name}) by {label}, "
                f"range {vmin:.2f}–{vmax:.2f} ({units}).\n"
                f"Values volume: #{vals_vol.id_string} (hidden)"
            )

    def _make_progress_cb(self, progressbar, prefix):
        """Return a callback(fraction 0..1) that drives a QProgressBar and
        also prints the percentage to the ChimeraX status line."""
        def cb(frac):
            pct = int(max(0.0, min(1.0, frac)) * 100)
            progressbar.setValue(pct)
            progressbar.setFormat(f"{prefix}  %p%")
            self.session.logger.status(f"{prefix} {pct}%")
            QApplication.processEvents()   # repaint mid-computation
        return cb

    def _matplotlib_palette(self, cmap_name, vmin, vmax, n=9):
        """Build a ChimeraX palette string sampled from a matplotlib colormap.

        Returns 'v1,#rrggbb:v2,#rrggbb:...' spanning vmin..vmax so the ChimeraX
        map coloring matches the matplotlib histogram colors.
        """
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        cmap = mcm.get_cmap(cmap_name)
        stops = []
        for i in range(n):
            frac = i / (n - 1)
            value = vmin + frac * (vmax - vmin)
            hex_color = mcolors.to_hex(cmap(frac))
            stops.append(f"{value:.4f},{hex_color}")
        return ":".join(stops)

    # ==================================================================
    # Shared — volume / structure list refresh
    # ==================================================================

    def _auto_fill_pixel_size(self):
        for combo in (self._map1_combo, self._map2_combo):
            vol = combo.currentData()
            if vol is not None:
                try:
                    step = float(vol.data.step[0])
                    if step > 0:
                        self._pix_spin.setValue(step)
                        self._sim_res_spin.setValue(2.0 * step)
                        return
                except Exception:
                    pass

    def _refresh_volume_list(self):
        from chimerax.map import Volume
        vols = self.session.models.list(type=Volume)

        for combo in (self._map1_combo, self._map2_combo):
            prev = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            sel = 0
            for idx, v in enumerate(vols):
                combo.addItem(f"#{v.id_string}  {v.name}", userData=v)
                if v is prev:
                    sel = idx
            combo.setCurrentIndex(sel)
            combo.blockSignals(False)

        prev_mask = self._mask_combo.currentData()
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        self._mask_combo.addItem("None", userData=None)
        sel = 0
        for idx, v in enumerate(vols):
            self._mask_combo.addItem(f"#{v.id_string}  {v.name}", userData=v)
            if v is prev_mask:
                sel = idx + 1
        self._mask_combo.setCurrentIndex(sel)
        self._mask_combo.blockSignals(False)

        self._auto_fill_pixel_size()

    def _refresh_structure_list(self):
        from chimerax.atomic import AtomicStructure
        structs = self.session.models.list(type=AtomicStructure)

        prev = self._struct_combo.currentData()
        self._struct_combo.blockSignals(True)
        self._struct_combo.clear()
        sel = 0
        for idx, s in enumerate(structs):
            self._struct_combo.addItem(f"#{s.id_string}  {s.name}", userData=s)
            if s is prev:
                sel = idx
        self._struct_combo.setCurrentIndex(sel)
        self._struct_combo.blockSignals(False)

    def _on_models_changed(self, *args):
        self._refresh_volume_list()
        self._refresh_structure_list()
        self._refresh_lr_volume_list()
        self._refresh_dr_volume_list()

    # ==================================================================
    # Tab 1 — FSC compute
    # ==================================================================

    def _on_compute(self):
        map1 = self._map1_combo.currentData()
        mask = self._mask_combo.currentData()
        pixel_size = self._pix_spin.value()

        if map1 is None:
            self.session.logger.error("Please open at least one volume first.")
            return

        if self._mode_map.isChecked():
            map2 = self._map2_combo.currentData()
            if map2 is None:
                self.session.logger.error("Please open at least two volumes first.")
                return
            if map1 is map2:
                self.session.logger.warning(
                    "Map 1 and Map 2 are the same model — results will be trivial."
                )
            self.compute_and_show(map1, map2, pixel_size, mask)
        else:
            structure = self._struct_combo.currentData()
            if structure is None:
                self.session.logger.error("Please open an atomic model first.")
                return
            self._compute_pdb_mode(map1, structure, pixel_size, mask,
                                   self._sim_res_spin.value())

    def _run_molmap(self, structure, ref_map, resolution):
        from chimerax.core.commands import run
        from chimerax.map import Volume

        if self._sim_map is not None:
            try:
                run(self.session, f"close #{self._sim_map.id_string}")
            except Exception:
                pass
            self._sim_map = None

        def new_volumes(before):
            after = set(self.session.models.list(type=Volume))
            return after - before

        before1 = set(self.session.models.list(type=Volume))
        run(self.session,
            f"molmap #{structure.id_string} {resolution} "
            f"onGrid #{ref_map.id_string}")
        created = new_volumes(before1)
        if not created:
            raise RuntimeError("molmap produced no output volume.")
        mol_vol = next(iter(created))

        before2 = set(self.session.models.list(type=Volume))
        run(self.session,
            f"volume resample #{mol_vol.id_string} onGrid #{ref_map.id_string}")
        resampled = new_volumes(before2)

        run(self.session, f"close #{mol_vol.id_string}")

        if not resampled:
            raise RuntimeError("volume resample produced no output.")
        return next(iter(resampled))

    def _compute_pdb_mode(self, map1, structure, pixel_size, mask, sim_resolution):
        self.session.logger.status("Running molmap …")
        try:
            sim_map = self._run_molmap(structure, map1, sim_resolution)
        except Exception as e:
            self.session.logger.error(f"molmap failed: {e}")
            return
        self._sim_map = sim_map
        self.session.logger.info(
            f"Simulated map: #{sim_map.id_string}  {sim_map.name}"
        )
        self.compute_and_show(map1, sim_map, pixel_size, mask)

    def compute_and_show(self, map1, map2, pixel_size, mask=None):
        """Public entry point — also called by the fsc command."""
        from .fsc_core import compute_fsc, find_resolution

        self.session.logger.status("Computing Global FSC …")
        self._compute_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        QApplication.processEvents()   # force the bar to paint before we block

        import time
        _t0 = time.perf_counter()

        try:
            vol1 = map1.data.matrix().astype(np.float32, copy=True)
            vol2 = map2.data.matrix().astype(np.float32, copy=True)
            mask_arr = (mask.data.matrix().astype(np.float32, copy=True)
                        if mask is not None else None)

            if vol1.shape != vol2.shape:
                self.session.logger.error(
                    f"Shape mismatch: {map1.name} {vol1.shape} vs {map2.name} {vol2.shape}"
                )
                return

            try:
                s1, s2 = map1.data.step, map2.data.step
                if s1 and s2 and not all(abs(a - b) < 1e-4 for a, b in zip(s1, s2)):
                    self.session.logger.warning(
                        f"Pixel sizes differ: {map1.name} {s1} vs {map2.name} {s2}. "
                        "Results may be unreliable."
                    )
            except Exception:
                pass

            threshold = self._thresh_spin.value()
            cb = self._make_progress_cb(self._progress, "Computing Global FSC…")
            fsc_values = compute_fsc(vol1, vol2, mask_arr, progress_callback=cb)
            resolution = find_resolution(fsc_values, pixel_size, threshold=threshold)

            self._fsc_values = fsc_values
            self._pixel_size = pixel_size

            self._update_plot(fsc_values, pixel_size, resolution, threshold)

            n = len(fsc_values)
            bins = np.linspace(0.0, 0.5, n + 1)
            centers = (bins[:-1] + bins[1:]) / 2.0
            with np.errstate(divide='ignore'):
                res_axis = np.where(centers > 0, pixel_size / centers, np.inf)
            lines = ["FSC table:",
                     f"  {'Shell':>5}  {'Freq':>8}  {'1/Å':>10}  {'Resolution (Å)':>15}  {'FSC':>7}"]
            for i in range(1, n):
                spatial = centers[i] / pixel_size
                res_str = f"{res_axis[i]:.3f}" if np.isfinite(res_axis[i]) else "inf"
                lines.append(
                    f"  {i:>5}  {centers[i]:>8.4f}  {spatial:>10.4f}  {res_str:>15}  {fsc_values[i]:>7.4f}"
                )
            self.session.logger.info("\n".join(lines))

            if resolution is not None:
                res_str = f"{resolution:.2f} Å"
                self._res_label.setText(
                    f"Resolution at FSC={threshold:.3f}:  {res_str}"
                )
                self.session.logger.info(
                    f"FSC resolution ({map1.name} vs {map2.name}): {res_str}"
                )
            else:
                self._res_label.setText(
                    f"Resolution at FSC={threshold:.3f}:  N/A (FSC > {threshold:.3f} throughout)"
                )

            elapsed = time.perf_counter() - _t0
            self.session.logger.info(f"FSC computed in {elapsed:.2f} s.")

        finally:
            self._compute_btn.setEnabled(True)
            self._progress.setVisible(False)
            self.session.logger.status("FSC done.")

    def _on_threshold_changed(self, value):
        if self._fsc_values is None or self._pixel_size is None:
            return
        from .fsc_core import find_resolution
        threshold = float(value)
        resolution = find_resolution(self._fsc_values, self._pixel_size,
                                     threshold=threshold)
        self._update_plot(self._fsc_values, self._pixel_size, resolution, threshold)
        if resolution is not None:
            self._res_label.setText(
                f"Resolution at FSC={threshold:.3f}:  {resolution:.2f} Å"
            )
        else:
            self._res_label.setText(f"Resolution at FSC={threshold:.3f}:  N/A")

    def _on_save_image(self):
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Save FSC plot", "fsc_plot.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
            self.session.logger.info(f"FSC plot saved to {path}")

    def _on_save_data(self):
        from Qt.QtWidgets import QFileDialog
        import csv
        if self._fsc_values is None:
            self.session.logger.error("No FSC data to save — run Compute first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Save FSC data", "fsc_data.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        n = len(self._fsc_values)
        bins = np.linspace(0.0, 0.5, n + 1)
        centers = (bins[:-1] + bins[1:]) / 2.0
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['shell', 'digital_freq', 'spatial_freq_inv_A',
                             'resolution_A', 'fsc'])
            for i in range(1, n):
                freq = centers[i]
                spatial = freq / self._pixel_size
                with np.errstate(divide='ignore'):
                    res = 1.0 / spatial if spatial > 0 else float('inf')
                writer.writerow([i, f"{freq:.6f}", f"{spatial:.6f}",
                                 f"{res:.4f}" if np.isfinite(res) else "inf",
                                 f"{self._fsc_values[i]:.6f}"])
        self.session.logger.info(f"FSC data saved to {path}")

    def _update_plot(self, fsc_values, pixel_size, resolution, threshold=0.143):
        n = len(fsc_values)
        bins = np.linspace(0.0, 0.5, n + 1)
        centers = (bins[:-1] + bins[1:]) / 2.0
        freq_axis = centers[1:] / pixel_size
        x_max = 0.5 / pixel_size

        ax = self._ax
        ax.cla()
        ax.set_xlabel("Spatial frequency (1/Å)", fontsize=9)
        ax.set_ylabel("FSC", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.25)

        ax.plot(freq_axis, fsc_values[1:],
                color='steelblue', linewidth=1.5, label='FSC')
        ax.axhline(threshold, color='tomato', linestyle='--',
                   linewidth=1, label=f'FSC={threshold:.3f}')
        ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.4)

        if resolution is not None and pixel_size > 0:
            freq_cross = 1.0 / resolution
            ax.axvline(freq_cross, color='dimgrey', linestyle=':',
                       linewidth=1, label=f'{resolution:.2f} Å')
            ax.text(freq_cross, 0.20, f'{resolution:.2f} Å',
                    ha='center', va='bottom', fontsize=9, color='dimgrey',
                    bbox=dict(facecolor='white', edgecolor='none',
                              boxstyle='round,pad=0.2', alpha=0.8))

        ax.legend(loc='upper left', fontsize=8)
        self._canvas.draw()

    # ==================================================================
    # Cleanup
    # ==================================================================

    def delete(self):
        for h in self._handlers:
            self.session.triggers.remove_handler(h)
        super().delete()
