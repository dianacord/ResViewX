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

        # Directional resolution (FSO) state
        self._dr_fso = None      # FSO curve data
        self._dr_dr = None       # (zenith, azimuth) resolution grid
        self._dr_view_handler = None
        self._dr_view_vol = None
        self._dr_view_mode = None
        self._dr_last_dir = None

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
        self._tabs = tabs

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

    # ==================================================================
    # Help and citation buttons, shared by all three tabs
    # ==================================================================

    HELP_TEXT = {
        "fsc": (
            "<b>Global FSC</b><br><br>"
            "Measures how well two half maps agree at each spatial frequency, "
            "and reports the resolution where that agreement falls below a "
            "threshold.<br><br>"
            "<b>Map vs Map</b> compares two half maps. <b>Map vs Atomic model</b> "
            "simulates a map from an atomic model with <tt>molmap</tt> and "
            "compares against that.<br><br>"
            "<b>Threshold</b> defaults to 0.143 for half maps and 0.5 for "
            "map-to-model, the usual conventions.<br>"
            "<b>Sim. resolution</b> (Map vs Atomic model) defaults to "
            "<b>Nyquist</b>, twice the pixel size \u2014 the finest detail the "
            "sampling can represent. Lower it to match your map\u2019s actual "
            "resolution for a like-for-like comparison.<br>"
            "<b>Pixel size</b> is read from the map header; it rescales every "
            "resolution, so check it is right.<br>"
            "<b>Mask</b> is optional, but an unmasked FSC includes solvent noise "
            "and usually reports a worse resolution.<br><br>"
            "Click anywhere on the plot to read off resolution, frequency and "
            "FSC at that point.<br><br>"
            "From the command line: <tt>fsc #1 #2 mask #3</tt>"),
        "local": (
            "<b>Local Resolution</b><br><br>"
            "Estimates resolution separately at every voxel, so you can see "
            "which parts of the map are well resolved and which are not. The "
            "colour varies across the structure and does not change as you "
            "rotate it.<br><br>"
            "<b>Map 1</b> and a <b>mask</b> are required; a second half map is "
            "optional and improves the noise estimate.<br>"
            "<b>Max resolution</b> is the low-resolution end where the sweep "
            "starts; it steps down towards Nyquist (2 x pixel size).<br>"
            "<b>Step size</b> is the resolution increment, minimum 0.25 A.<br>"
            "<b>Significance</b> is the confidence level of the signal/noise "
            "test.<br><br>"
            "Set the display range and press Enter to rescale the histogram. "
            "<b>Color Map in ChimeraX</b> paints a copy of the map, leaving "
            "your original untouched."),
        "directional": (
            "<b>Directional Resolution</b><br><br>"
            "Measures resolution as a function of direction, which reveals "
            "anisotropy: a map can be sharp in one direction and blurred in "
            "another, usually because particles adopted preferred "
            "orientations.<br><br>"
            "<b>Two different half maps</b> are required; a mask is optional "
            "but restricts the measurement to the specimen.<br>"
            "<b>Cone angle</b> is the half-angle of the Fourier cone sampled "
            "per direction. <b>FSC threshold</b> is what a direction must "
            "exceed to count as occupied.<br><br>"
            "<b>FSO curve</b>: the fraction of directions still above the "
            "threshold in each shell. It falls from 1 to 0, and the width of "
            "that fall is the anisotropy &mdash; wider means more anisotropic.<br>"
            "<b>DR polar</b>: resolution per direction, azimuth around the "
            "circle and zenith as the radius.<br><br>"
            "<b>Color by viewing direction</b> paints the whole map one colour, "
            "the resolution along the direction you are looking from.<br>"
            "<b>Color by directional reliability</b> shows the same measurement "
            "inverted, on an arbitrary 0 to 1 scale: purple (1) is the most "
            "reliable view and red (0) the least. The inversion is deliberate "
            "\u2014 the resolution limiting a given view lies perpendicular to "
            "it, so the direction with the worst measured resolution is the one "
            "you can trust most. Keeping it off the \u00c5ngstr\u00f6m scale avoids "
            "implying that "
            "the Angstrom value belongs to the surface you are looking at: the "
            "resolution limiting a given view lies perpendicular to it."),
    }

    CITATIONS = {
        "fsc": [
            ("ResViewX (this plugin)",
             "Cordovez, D., Vilas, J.L. &amp; S\u00e1nchez-Garc\u00eda, R. (in preparation). ResViewX: global, local and directional resolution analysis of cryo-EM maps within UCSF ChimeraX."),
            ("FSC 0.143 resolution criterion",
             "Rosenthal, P.B. &amp; Henderson, R. (2003). Optimal determination of "
             "particle orientation, absolute hand, and contrast loss in "
             "single-particle electron cryomicroscopy. J Mol Biol 333(4), "
             "721-745. doi:10.1016/j.jmb.2003.07.013"),
            ("Simulated maps from atomic models (molmap) and the host application",
             "Meng, E.C. et al. (2023). UCSF ChimeraX: Tools for structure "
             "building and analysis. Protein Sci 32(11), e4792. "
             "doi:10.1002/pro.4792"),
        ],
        "local": [
            ("ResViewX (this plugin)",
             "Cordovez, D., Vilas, J.L. &amp; S\u00e1nchez-Garc\u00eda, R. (in preparation). ResViewX: global, local and directional resolution analysis of cryo-EM maps within UCSF ChimeraX."),
            ("MonoRes local resolution",
             "Vilas, J.L. et al. (2018). MonoRes: Automatic and Accurate "
             "Estimation of Local Resolution for Electron Microscopy Maps. "
             "Structure 26(2), 337-344.e4. doi:10.1016/j.str.2017.12.018"),
            ("Host application",
             "Meng, E.C. et al. (2023). UCSF ChimeraX: Tools for structure "
             "building and analysis. Protein Sci 32(11), e4792. "
             "doi:10.1002/pro.4792"),
        ],
        "directional": [
            ("ResViewX (this plugin)",
             "Cordovez, D., Vilas, J.L. &amp; S\u00e1nchez-Garc\u00eda, R. (in preparation). ResViewX: global, local and directional resolution analysis of cryo-EM maps within UCSF ChimeraX."),
            ("Fourier Shell Occupancy and the Bingham anisotropy test",
             "Vilas, J.L., Tagare, H.D., Vargas, J., Carazo, J.M. &amp; Sorzano, "
             "C.O.S. (2023). New measures of anisotropy of cryo-EM maps. "
             "Nat Methods 20, 1021-1024. doi:10.1038/s41592-023-01874-3"),
            ("FSC 0.143 resolution criterion",
             "Rosenthal, P.B. &amp; Henderson, R. (2003). J Mol Biol 333(4), "
             "721-745. doi:10.1016/j.jmb.2003.07.013"),
            ("Host application",
             "Meng, E.C. et al. (2023). UCSF ChimeraX: Tools for structure "
             "building and analysis. Protein Sci 32(11), e4792. "
             "doi:10.1002/pro.4792"),
        ],
    }

    # Short explanations for the "?" button beside each input. Wording follows
    # the Scipion/Xmipp protocol help where the parameter is the same one.
    FIELD_HELP = {
        # Global FSC
        "fsc_map1": "The map being analysed. In Map vs Map this is the first of "
                    "two half maps reconstructed from independent halves of the "
                    "particle set; in Map vs Atomic model it is the only map "
                    "needed, and is compared against the model.",
        "fsc_map2": "Second half map. Must be on the same grid as the first.",
        "fsc_struct": "Atomic model to compare the map against.",
        "fsc_simres": "Resolution of the map simulated from the atomic model. "
                      "Defaults to Nyquist, twice the pixel size.",
        "fsc_mask": "Optional. Restricts the comparison to the specimen. Without a "
                    "mask the surrounding solvent noise is included and the "
                    "reported resolution is usually worse.",
        "fsc_pixel": "Pixel (voxel) size in Angstrom, read from the map header. "
                     "Every reported resolution scales with this, so check it.",
        "fsc_thresh": "Threshold for the FSC. Defaults to the standard 0.143 for "
                      "Map vs Map, and to 0.5 for Map vs Atomic model, which is "
                      "the convention for map-to-model comparison. Other common "
                      "thresholds are 0.3 and 0.5.",
        # Local resolution
        "lr_map1": "Select the first half of a volume for determining its local "
                   "resolution.",
        "lr_map2": "Optional second half map. The noise estimate is made from the "
                   "difference between the two halves, so providing it improves "
                   "the estimate.",
        "lr_mask": "The mask determines which points are specimen and which are not.",
        "lr_pixel": "Pixel (voxel) size in Angstrom, read from the map header.",
        "lr_maxres": "The range of resolutions to be analysed. The local resolution "
                     "is estimated from low to high, starting here and stepping "
                     "down towards Nyquist.",
        "lr_step": "The resolution will be sought in steps of this value. Clamped "
                   "to a minimum of 0.25 Angstrom.",
        "lr_signif": "Resolution is computed using hypothesis tests; this value "
                     "determines the significance of that test.",
        # Directional resolution
        "dr_half1": "Select one map for determining the directional FSC resolution.",
        "dr_half2": "Select the second map for determining the directional FSC "
                    "resolution.",
        "dr_mask": "Optional. The mask determines which points are specimen and "
                   "which are not.",
        "dr_pixel": "Pixel (voxel) size in Angstrom, read from the map header.",
        "dr_cone": "Angle between the axis of the cone and the generatrix. "
                   "17 degrees is the default (see Nat Methods, J.L. Vilas 2023).",
        "dr_thresh": "Threshold for the FSC. By default the standard 0.143. Other "
                     "common thresholds are 0.5 and 0.3.",
        "dr_fast": "321 directions are sampled by default. The 81-direction set is "
                   "about four times faster but samples the sphere more coarsely.",
    }

    def _with_help(self, widget, key):
        """Put a small '?' beside an input. Hovering shows the text, clicking
        opens it in a box for the longer ones."""
        from Qt.QtWidgets import QToolButton
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(widget, 1)
        btn = QToolButton()
        btn.setText("?")
        btn.setAutoRaise(True)
        btn.setToolTip(self.FIELD_HELP[key])
        btn.setFixedWidth(18)
        btn.clicked.connect(lambda: self._show_field_help(key))
        row.addWidget(btn, 0)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    def _show_field_help(self, key):
        from Qt.QtWidgets import QMessageBox
        box = QMessageBox(self.tool_window.ui_area)
        box.setWindowTitle("What is this?")
        box.setText(self.FIELD_HELP[key])
        box.exec()

    def _make_help_row(self, key, title):
        """A Help and a Citations button, used at the top of every tab."""
        row = QHBoxLayout()
        help_btn = QPushButton("Help")
        help_btn.setToolTip("What this tab does and what the inputs mean")
        help_btn.clicked.connect(lambda: self._show_help(key, title))
        cite_btn = QPushButton("Citations")
        cite_btn.setToolTip("Papers describing the algorithms used here")
        cite_btn.clicked.connect(lambda: self._show_citations(key, title))
        row.addStretch()
        row.addWidget(help_btn)
        row.addWidget(cite_btn)
        return row

    def _show_help(self, key, title):
        from Qt.QtWidgets import QMessageBox
        box = QMessageBox(self.tool_window.ui_area)
        box.setWindowTitle(f"{title} \u2014 help")
        box.setTextFormat(Qt.RichText)
        box.setText(self.HELP_TEXT[key])
        box.exec()

    def _show_citations(self, key, title):
        """Show the papers behind this tab, with a button to copy them in a
        form that can be pasted straight into a reference list."""
        from Qt.QtWidgets import QMessageBox
        entries = self.CITATIONS[key]
        html = "<b>Please cite the methods you use.</b><br><br>" + "<br><br>".join(
            f"<i>{what}</i><br>{ref}" for what, ref in entries)
        plain = "\n\n".join(ref.replace("&amp;", "&") for _what, ref in entries)

        box = QMessageBox(self.tool_window.ui_area)
        box.setWindowTitle(f"{title} \u2014 citations")
        box.setTextFormat(Qt.RichText)
        box.setText(html)
        copy_btn = box.addButton("Copy citations", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is copy_btn:
            from Qt.QtWidgets import QApplication as _QApp
            _QApp.clipboard().setText(plain)
            self.session.logger.status("Citations copied to the clipboard.")

        annotated = "\n\n".join(
            f"{what}:\n  {ref}".replace("&amp;", "&") for what, ref in entries)
        self.session.logger.info(f"{title} \u2014 algorithms used:\n{annotated}")

    def _build_fsc_tab(self, layout):
        layout.addLayout(self._make_help_row('fsc', 'Global FSC'))

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

        form.addRow("Map 1:", self._with_help(self._map1_combo, "fsc_map1"))
        # Keep the holders: hiding only the inner widget would leave its "?"
        # button floating on its own when the mode changes.
        self._map2_holder = self._with_help(self._map2_combo, "fsc_map2")
        self._struct_holder = self._with_help(self._struct_combo, "fsc_struct")
        self._sim_res_holder = self._with_help(self._sim_res_spin, "fsc_simres")
        form.addRow(self._map2_label, self._map2_holder)
        form.addRow(self._struct_label, self._struct_holder)
        form.addRow(self._sim_res_label, self._sim_res_holder)
        form.addRow("Mask (opt.):", self._with_help(self._mask_combo, "fsc_mask"))
        form.addRow("Pixel size:", self._with_help(self._pix_spin, "fsc_pixel"))
        form.addRow("Threshold:", self._with_help(self._thresh_spin, "fsc_thresh"))
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
        self._map2_holder.setVisible(map_mode)
        self._struct_label.setVisible(not map_mode)
        self._struct_holder.setVisible(not map_mode)
        self._sim_res_label.setVisible(not map_mode)
        self._sim_res_holder.setVisible(not map_mode)
        self._thresh_spin.setValue(0.143 if map_mode else 0.5)

    def _make_fsc_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        self._fig = Figure(figsize=(5, 3.5), constrained_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._draw_empty_fsc_axes()
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._fsc_click_marker = None
        self._fsc_curve = None
        self._fsc_last_idx = None
        self._canvas.mpl_connect('button_press_event', self._on_fsc_point)
        self._canvas.mpl_connect('motion_notify_event', self._on_fsc_point)
        return self._canvas

    def _draw_empty_fsc_axes(self):
        ax = self._ax
        ax.cla()
        ax.set_xlabel("Resolution (Å)", fontsize=9)
        ax.set_ylabel("FSC", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.143, color='tomato', linestyle='--', linewidth=1,
                   label='0.143 threshold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    def _on_fsc_point(self, event):
        """Read the FSC off the curve. The reported value is taken from the
        plotted data at the nearest frequency, not from where the pointer
        happens to sit, so it always lies on the line."""
        curve = getattr(self, "_fsc_curve", None)
        if curve is None or event.inaxes is not self._ax or event.xdata is None:
            return
        freqs, values = curve
        if len(freqs) == 0:
            return

        idx = int(np.argmin(np.abs(freqs - float(event.xdata))))
        if idx == getattr(self, "_fsc_last_idx", None):
            return                      # same point, nothing to redraw
        self._fsc_last_idx = idx

        freq = float(freqs[idx])
        fsc = float(values[idx])
        if freq <= 0:
            return
        res = 1.0 / freq

        # Replace the previous readout rather than stacking them
        marker = getattr(self, "_fsc_click_marker", None)
        if marker is not None:
            for artist in marker:
                try:
                    artist.remove()
                except Exception:
                    pass

        pt = self._ax.plot([freq], [fsc], marker='o', markersize=5,
                           color='black', zorder=5)[0]

        # Flip the label inwards near an edge so the box stays over the plot
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        right = freq > x0 + 0.72 * (x1 - x0)
        high = fsc > y0 + 0.78 * (y1 - y0)
        dx, ha = (-8, 'right') if right else (8, 'left')
        dy, va = (-10, 'top') if high else (8, 'bottom')

        txt = self._ax.annotate(
            f"{res:.2f} \u00c5  ({freq:.4f} 1/\u00c5)\nFSC = {fsc:.3f}",
            xy=(freq, fsc), xytext=(dx, dy), textcoords='offset points',
            ha=ha, va=va, fontsize=8, zorder=6, annotation_clip=True,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='grey', alpha=0.95))
        # Keep the label out of the layout calculation: constrained_layout was
        # resizing the axes whenever the box reached near an edge.
        txt.set_in_layout(False)
        self._fsc_click_marker = (pt, txt)
        self._canvas.draw_idle()
        self.session.logger.status(
            f"FSC at {res:.2f} \u00c5 ({freq:.4f} 1/\u00c5): {fsc:.3f}")

    def _build_localres_tab(self, layout):
        layout.addLayout(self._make_help_row('local', 'Local Resolution'))
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

        # Required inputs first, optional second half map after
        form.addRow("Map 1:", self._with_help(self._lr_map1_combo, "lr_map1"))
        form.addRow("Mask:", self._with_help(self._lr_mask_combo, "lr_mask"))
        form.addRow("Map 2 (opt.):", self._with_help(self._lr_map2_combo, "lr_map2"))
        form.addRow("Pixel size:", self._with_help(self._lr_pix_spin, "lr_pixel"))
        form.addRow("Max resolution:", self._with_help(self._lr_maxres_spin, "lr_maxres"))
        form.addRow("Step size:", self._with_help(self._lr_step_spin, "lr_step"))
        form.addRow("Significance:", self._with_help(self._lr_signif_spin, "lr_signif"))
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
        # editingFinished fires on Enter or when the field loses focus, so the
        # histogram follows the range without a separate button press.
        for spin in (self._lr_res_min_spin, self._lr_res_max_spin):
            spin.setKeyboardTracking(False)
            spin.editingFinished.connect(self._on_lr_update_histogram)
        range_row.addWidget(self._lr_res_min_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._lr_res_max_spin)

        self._lr_cmap_combo = QComboBox()
        for label, key in (("viridis", "viridis"), ("blue-white-red", "bwr")):
            self._lr_cmap_combo.addItem(label, userData=key)
        self._lr_cmap_combo.currentIndexChanged.connect(self._on_lr_update_histogram)
        range_row.addWidget(QLabel("Colours:"))
        range_row.addWidget(self._lr_cmap_combo)
        range_row.addStretch()
        layout.addLayout(range_row)

        layout.addWidget(self._make_lr_canvas())

        lr_btn_row = QHBoxLayout()
        self._lr_color_btn = QPushButton("Color Map in ChimeraX")
        self._lr_color_btn.clicked.connect(self._on_lr_color_map)
        self._lr_color_btn.setEnabled(False)
        self._lr_export_btn = QPushButton("Export histogram")
        self._lr_export_btn.clicked.connect(self._on_lr_export_histogram)
        self._lr_export_btn.setEnabled(False)
        lr_btn_row.addWidget(self._lr_color_btn)
        lr_btn_row.addWidget(self._lr_export_btn)
        layout.addLayout(lr_btn_row)

    def _lr_cmap(self):
        """Colormap chosen for the local resolution tab."""
        return self._lr_cmap_combo.currentData() or "viridis"

    def _on_lr_export_histogram(self):
        """Save the histogram as an image, or the underlying values as CSV."""
        from Qt.QtWidgets import QFileDialog
        import csv
        if self._lr_output is None:
            self.session.logger.error("Run Local Resolution first.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, "Export histogram",
            "local_resolution_histogram.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg);;CSV data (*.csv)")
        if not path:
            return
        if path.lower().endswith(".csv") or "CSV" in (selected or ""):
            vmin = self._lr_res_min_spin.value()
            vmax = self._lr_res_max_spin.value()
            flat = self._lr_output.ravel()
            flat = flat[np.isfinite(flat)]
            counts, edges = np.histogram(
                flat[(flat >= vmin) & (flat <= vmax)], bins=60,
                range=(vmin, vmax))
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin_start_A", "bin_end_A", "bin_center_A", "voxel_count"])
                for i, c in enumerate(counts):
                    w.writerow([f"{edges[i]:.4f}", f"{edges[i+1]:.4f}",
                                f"{0.5*(edges[i]+edges[i+1]):.4f}", int(c)])
            self.session.logger.info(f"Histogram data exported to {path}")
        else:
            self._lr_fig.savefig(path, dpi=150, bbox_inches="tight")
            self.session.logger.info(f"Histogram image exported to {path}")

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

        self._lr_color_btn.setEnabled(True)
        self._lr_export_btn.setEnabled(True)
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
                                    "Local resolution (Å)", self._lr_cmap())

    def _draw_metric_histogram(self, ax, cax, canvas, output, vmin, vmax, xlabel,
                               cmap_name='rainbow'):
        """Shared histogram + matching colorbar used by the Local and
        Directional tabs. Bars are coloured by value using `cmap_name`, which
        must match whatever the map colouring and colour key use."""
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors

        if vmax <= vmin:
            vmax = vmin + 1e-6

        flat = output.ravel()
        flat = flat[np.isfinite(flat)]
        in_range = flat[(flat >= vmin) & (flat <= vmax)]

        cmap = mcm.get_cmap(cmap_name)
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
            prev_attr="_lr_vol", key_title="Local resolution (Å)",
            cmap_name=self._lr_cmap())
        if result is not None:
            map_copy, vals_vol = result
            self.session.logger.info(
                f"Colored map #{map_copy.id_string} ({map1.name}) "
                f"by local resolution, range {res_min:.1f}–{res_max:.1f} Å.\n"
                f"Resolution values volume: #{vals_vol.id_string} (hidden)"
            )

    def _make_surface_copy(self, map1, copy_name):
        """Copy `map1` as a new volume shown as a surface, so the original is
        never modified. The copy is built with its final name because Volume
        Viewer reads a map's name only when the model is added and ignores
        later renames. Returns the copy, or None on failure."""
        from chimerax.map_data import ArrayGridData
        from chimerax.map import volume_from_grid_data
        from chimerax.core.commands import run

        copy_grid = ArrayGridData(map1.data.matrix().copy(),
                                  origin=map1.data.origin,
                                  step=map1.data.step, name=copy_name)
        map_copy = volume_from_grid_data(copy_grid, self.session, open_model=True)
        if map_copy is None:
            self.session.logger.error("Could not create a copy of the map.")
            return None
        map_copy.name = copy_name

        # The surface needs real geometry before it can be coloured. Use the
        # original's contour level if it has one, else derive one.
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

        # ChimeraX builds big surfaces in a background thread; force them to
        # finish now so the geometry exists when we colour it.
        try:
            ro = map_copy.rendering_options
            for s in map_copy.surfaces:
                s.set_level(s.level, use_thread=False)
                s.update_surface(ro)
        except Exception as e:
            self.session.logger.warning(f"Surface pre-compute skipped: {e}")
        return map_copy

    def _color_map_by_array(self, map1, array, vmin, vmax,
                            values_name, copy_name, prev_attr, key_title,
                            cmap_name='rainbow'):
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

        map_copy = self._make_surface_copy(map1, copy_name)
        if map_copy is None:
            return None
        setattr(self, prev_attr + "_copy", map_copy)

        # Palette sampled from matplotlib 'rainbow' so the map matches the histogram
        palette = self._matplotlib_palette(cmap_name, vmin, vmax, n=32)
        run(self.session,
            f"color sample #{map_copy.id_string} map #{vals_vol.id_string} "
            f"palette {palette} range {vmin},{vmax}")

        # Hide the raw values volume so it doesn't obscure the colored map
        run(self.session, f"hide #{vals_vol.id_string} models")

        self._show_color_key(vmin, vmax, key_title, cmap_name=cmap_name)
        return map_copy, vals_vol

    def _show_color_key(self, vmin, vmax, title, n=32, n_labels=5,
                        cmap_name='rainbow'):
        """Draw the same matplotlib 'rainbow' colorbar in the 3D view (bottom
        left) so the color scale is visible while rotating the structure."""
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        from chimerax.core.commands import run

        cmap = mcm.get_cmap(cmap_name)
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
    # Tab 3 — Directional Resolution (FSO)
    # ==================================================================

    def _build_directional_tab(self, layout):
        layout.addLayout(
            self._make_help_row('directional', 'Directional Resolution'))
        hint = QLabel(
            "Two different half maps. A mask is optional.")
        hint.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(4)

        self._dr_half1_combo = QComboBox()
        self._dr_half2_combo = QComboBox()
        self._dr_mask_combo = QComboBox()
        for combo in (self._dr_half1_combo, self._dr_half2_combo, self._dr_mask_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._dr_half1_combo.currentIndexChanged.connect(self._dr_auto_fill_pixel_size)

        self._dr_pix_spin = QDoubleSpinBox()
        self._dr_pix_spin.setRange(0.001, 100.0)
        self._dr_pix_spin.setDecimals(4)
        self._dr_pix_spin.setSingleStep(0.001)
        self._dr_pix_spin.setValue(1.0)
        self._dr_pix_spin.setSuffix("  \u00c5/vox")

        self._dr_cone_spin = QDoubleSpinBox()
        self._dr_cone_spin.setRange(1.0, 90.0)
        self._dr_cone_spin.setDecimals(1)
        self._dr_cone_spin.setValue(17.0)
        self._dr_cone_spin.setSuffix(" \u00b0")
        self._dr_cone_spin.setToolTip("Half-angle of the Fourier cone used per direction")

        self._dr_thresh_spin = QDoubleSpinBox()
        self._dr_thresh_spin.setRange(0.001, 1.0)
        self._dr_thresh_spin.setDecimals(3)
        self._dr_thresh_spin.setSingleStep(0.001)
        self._dr_thresh_spin.setValue(0.143)
        self._dr_thresh_spin.setToolTip(
            "FSC threshold a direction must exceed to count as occupied")

        form.addRow("Half map 1:", self._with_help(self._dr_half1_combo, "dr_half1"))
        form.addRow("Half map 2:", self._with_help(self._dr_half2_combo, "dr_half2"))
        form.addRow("Mask (opt.):", self._with_help(self._dr_mask_combo, "dr_mask"))
        form.addRow("Pixel size:", self._with_help(self._dr_pix_spin, "dr_pixel"))
        self._dr_fast_check = QCheckBox("Fast (81 directions instead of 321)")
        self._dr_fast_check.setToolTip(
            "321 directions by default. The 81-direction set is about 4x "
            "faster but samples the sphere more coarsely.")

        form.addRow("Cone angle:", self._with_help(self._dr_cone_spin, "dr_cone"))
        form.addRow("FSC threshold:", self._with_help(self._dr_thresh_spin, "dr_thresh"))
        form.addRow("", self._with_help(self._dr_fast_check, "dr_fast"))
        layout.addLayout(form)

        self._dr_compute_btn = QPushButton("Compute Directional Resolution")
        self._dr_compute_btn.clicked.connect(self._on_dr_compute)
        layout.addWidget(self._dr_compute_btn)

        self._dr_progress = QProgressBar()
        self._dr_progress.setRange(0, 100)
        self._dr_progress.setTextVisible(True)
        self._dr_progress.setFormat("Computing Directional Resolution\u2026  %p%")
        self._dr_progress.setVisible(False)
        layout.addWidget(self._dr_progress)

        self._dr_status_label = QLabel("")
        self._dr_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._dr_status_label)

        # --- 2D plots, one per sub-tab -----------------------------------
        plot_tabs = QTabWidget()

        fso_page = QWidget()
        fso_layout = QVBoxLayout()
        fso_layout.setContentsMargins(0, 0, 0, 0)
        fso_layout.addWidget(self._make_fso_canvas())
        fso_page.setLayout(fso_layout)
        plot_tabs.addTab(fso_page, "FSO curve")

        polar_page = QWidget()
        polar_layout = QVBoxLayout()
        polar_layout.setContentsMargins(0, 0, 0, 0)
        polar_layout.addWidget(self._make_polar_canvas())
        polar_page.setLayout(polar_layout)
        plot_tabs.addTab(polar_page, "DR polar")

        layout.addWidget(plot_tabs)

        # --- 3D colouring by viewing direction ---------------------------
        self._dr_view_check = QCheckBox(
            "Color map by viewing direction (updates as you rotate)")
        self._dr_view_check.setToolTip(
            "Paints the whole map one colour: the directional resolution along "
            "the current viewing direction, using the DR polar colour scale.")
        self._dr_view_check.setEnabled(False)
        self._dr_view_check.toggled.connect(self._on_dr_view_res_toggled)
        layout.addWidget(self._dr_view_check)

        self._dr_view_rel_check = QCheckBox(
            "Color map by directional reliability (arbitrary 0 to 1 scale)")
        self._dr_view_rel_check.setToolTip(
            "Same per-direction measurement, inverted, on an arbitrary 0 to 1 "
            "scale: 1 is the most reliable view (purple), 0 the least (red). "
            "It is inverted because the resolution limiting what you see lies "
            "perpendicular to the direction you look along, so the direction "
            "whose measured resolution is worst is the one you can trust most. "
            "Deliberately not in \u00c5ngstr\u00f6m, so the number is not read as "
            "the resolution of the surface in front of you.")
        self._dr_view_rel_check.setEnabled(False)
        self._dr_view_rel_check.toggled.connect(self._on_dr_view_rel_toggled)
        layout.addWidget(self._dr_view_rel_check)

        self._dr_view_label = QLabel("")
        self._dr_view_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._dr_view_label)

    def _make_fso_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        self._fso_fig = Figure(figsize=(5, 3.5), constrained_layout=True)
        self._fso_ax = self._fso_fig.add_subplot(111)
        self._fso_canvas = FigureCanvas(self._fso_fig)
        self._fso_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draw_empty_fso_axes()
        return self._fso_canvas

    def _draw_empty_fso_axes(self):
        ax = self._fso_ax
        ax.cla()
        ax.set_xlabel("Resolution (Å)", fontsize=9)
        ax.set_ylabel("FSO (a.u.)", fontsize=9)
        ax.set_ylim(-0.1, 1.1)
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='grey')
        ax.grid(True, alpha=0.25)
        self._fso_canvas.draw()

    def _dr_draw_fso(self):
        """FSO curve (green) with the Bingham p-value (red dashed) and the
        anisotropy band shaded green between the FSO=0.9 and FSO=0.1 crossings.
        X axis is spatial frequency, labelled in Å."""
        from matplotlib.ticker import FuncFormatter
        data = getattr(self, "_dr_fso", None)
        if not data:
            return
        freq, fso, bingham = data["freq"], data["fso"], data["bingham"]
        cr = data["crossings"]

        ax = self._fso_ax
        ax.cla()
        ax.plot(freq, fso, 'g', linewidth=1.5, label='FSO')
        ax.plot(freq, bingham, 'r--', linewidth=1, label='Bingham p-value')
        ax.hlines([0.1, 0.5, 0.9], freq[0], freq[-1],
                  colors='k', linestyles='dashed', linewidth=0.8)

        # Anisotropy band: the wider this is, the more anisotropic the map
        r09, r01 = cr.get(0.9), cr.get(0.1)
        if r09 and r01:
            ax.axvspan(1.0 / r09, 1.0 / r01, alpha=0.3, color='green')

        lines = [f"{thr} → {cr[thr]:.2f} Å" for thr in (0.9, 0.5, 0.1)
                 if cr.get(thr)]
        if r09 and r01:
            lines.append(f"band {r09:.2f}–{r01:.2f} Å")
        if lines:
            ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, p: f"{1.0 / v:.1f}" if v else ""))
        ax.set_xlim(freq[0], freq[-1])
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Resolution (Å)", fontsize=9)
        ax.set_ylabel("FSO (a.u.)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=8)
        self._fso_canvas.draw()

    def _make_polar_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        self._polar_fig = Figure(figsize=(5, 4), constrained_layout=True)
        self._polar_ax = self._polar_fig.add_subplot(111, projection='polar')
        self._polar_cbar = None
        self._polar_canvas = FigureCanvas(self._polar_fig)
        self._polar_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draw_empty_polar_axes()
        return self._polar_canvas

    def _draw_empty_polar_axes(self):
        ax = self._polar_ax
        ax.cla()
        ax.set_title("No data yet", fontsize=10, color='grey')
        self._polar_canvas.draw()

    def _dr_draw_polar(self):
        """Directional resolution as a function of direction: azimuth around
        the circle, zenith angle as the radius, colour = resolution in Å."""
        grid = getattr(self, "_dr_dr", None)
        if grid is None:
            return
        n_zen, n_azi = grid.shape
        zeniths = np.linspace(0, 90, n_zen)
        azimuths = np.radians(np.linspace(0, 360, n_azi, endpoint=False))
        theta, r = np.meshgrid(azimuths, zeniths)

        if not np.isfinite(grid).any():
            ax = self._polar_ax
            ax.cla()
            ax.set_title("No directional resolution to plot", fontsize=10,
                         color='grey')
            self._polar_canvas.draw()
            return
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        if vmax <= vmin:
            vmax = vmin + 1e-3

        ax = self._polar_ax
        ax.cla()
        if self._polar_cbar is not None:
            try:
                self._polar_cbar.remove()
            except Exception:
                pass
            self._polar_cbar = None

        levels = np.linspace(vmin, vmax, 25)
        pc = ax.contourf(theta, r, grid, levels=levels,
                         cmap='rainbow', extend='max')
        ax.set_theta_zero_location('E')
        # Radial ticks between spokes so they clear the gridlines
        ax.set_rlabel_position(112.5)
        ax.set_title("Directional resolution (\u00c5)", fontsize=10, pad=12)
        # Name both polar axes in one caption below the plot. Placing them
        # inside the axes overlapped the angle ticks and the title.
        ax.set_xlabel("angle = rot (\u00b0)    \u00b7    radius = tilt (\u00b0)",
                      fontsize=8, color='dimgrey', labelpad=10)
        self._polar_cbar = self._polar_fig.colorbar(pc, ax=ax, pad=0.1)
        self._polar_cbar.set_label("Resolution (Å)", fontsize=9)
        self._polar_canvas.draw()

    # ---- 3D: colour the whole map by the current viewing direction ----

    def _on_dr_view_res_toggled(self, checked):
        """Colour by directional resolution in Angstrom."""
        if checked and self._dr_view_rel_check.isChecked():
            self._dr_view_rel_check.blockSignals(True)
            self._dr_view_rel_check.setChecked(False)
            self._dr_view_rel_check.blockSignals(False)
        self._dr_set_view_mode('resolution' if checked else None,
                               self._dr_view_check)

    def _on_dr_view_rel_toggled(self, checked):
        """Colour by directional reliability: the same measurement on an
        arbitrary 0..1 scale, red meaning not reliable at all."""
        if checked and self._dr_view_check.isChecked():
            self._dr_view_check.blockSignals(True)
            self._dr_view_check.setChecked(False)
            self._dr_view_check.blockSignals(False)
        self._dr_set_view_mode('reliability' if checked else None,
                               self._dr_view_rel_check)

    def _dr_set_view_mode(self, mode, source_check):
        """Start or stop colouring the map by the current viewing direction.
        `mode` is 'resolution', 'reliability', or None to switch off."""
        if mode is not None:
            if getattr(self, "_dr_dr", None) is None:
                source_check.blockSignals(True)
                source_check.setChecked(False)
                source_check.blockSignals(False)
                self.session.logger.error("Run Directional Resolution first.")
                return
            src_map = self._dr_half1_combo.currentData()
            if src_map is None:
                source_check.blockSignals(True)
                source_check.setChecked(False)
                source_check.blockSignals(False)
                return
            # Work on a copy, like the Local Resolution tab, so the user's own
            # half map is never recoloured.
            self._dr_stop_view_colouring()
            copy_name = ("directional reliability" if mode == 'reliability'
                         else "directional resolution")
            vol = self._make_surface_copy(src_map, copy_name)
            if vol is None:
                source_check.blockSignals(True)
                source_check.setChecked(False)
                source_check.blockSignals(False)
                return
            self._dr_view_mode = mode
            self._dr_view_vol = vol
            self._dr_last_dir = None
            self._dr_view_handler = self.session.triggers.add_handler(
                'new frame', self._dr_view_frame_cb)
            grid = self._dr_dr
            if mode == 'reliability':
                # Arbitrary 0..1 scale with a reversed colormap, so 0 (not
                # reliable at all, i.e. the worst resolution) shows as red.
                self._show_color_key(0.0, 1.0,
                                     "Directional reliability (0\u20131)",
                                     cmap_name='rainbow_r')
            else:
                self._show_color_key(float(np.nanmin(grid)),
                                     float(np.nanmax(grid)),
                                     "Directional resolution (\u00c5)")
            self._dr_view_frame_cb(None, None)   # paint immediately
        else:
            self._dr_stop_view_colouring()

    def _dr_stop_view_colouring(self):
        h = getattr(self, "_dr_view_handler", None)
        if h is not None:
            self.session.triggers.remove_handler(h)
            self._dr_view_handler = None
        self._dr_view_mode = None
        self._dr_close_view_copy()
        self._clear_view_readout()
        self._dr_view_label.setText("")

    def _dr_close_view_copy(self):
        from chimerax.core.commands import run
        vol = getattr(self, "_dr_view_vol", None)
        if vol is not None:
            try:
                if not vol.deleted:
                    run(self.session, f"close #{vol.id_string}")
            except Exception:
                pass
            self._dr_view_vol = None

    def _dr_view_frame_cb(self, trigger_name, data):
        """Each frame: if the camera moved, recolour the map for the new
        viewing direction and report that direction."""
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        from .directional_core import direction_resolution, direction_angles

        vol = getattr(self, "_dr_view_vol", None)
        grid = getattr(self, "_dr_dr", None)
        if vol is None or grid is None or vol.deleted:
            return
        try:
            view_dir = self.session.main_view.camera.view_direction()
            d = vol.scene_position.inverse().transform_vector(view_dir)
        except Exception:
            return

        d = np.asarray(d, dtype=float)
        last = getattr(self, "_dr_last_dir", None)
        if last is not None and float(np.abs(d - last).max()) < 1e-4:
            return                      # camera hasn't really moved
        self._dr_last_dir = d

        res = direction_resolution(grid, d)
        angles = direction_angles(d)
        if res is None or angles is None:
            return
        azimuth, zenith = angles

        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        span = (vmax - vmin) or 1e-9
        mode = getattr(self, "_dr_view_mode", 'resolution')

        if mode == 'reliability':
            # Inverted on purpose. The resolution limiting what you see lies
            # perpendicular to the direction you look along, so the direction
            # whose measured resolution is worst corresponds to the most
            # reliable view. Worst measured resolution -> 1 (purple, most
            # reliable); best measured resolution -> 0 (red, not reliable).
            reliability = (res - vmin) / span
            frac = reliability
            cmap_name = 'rainbow_r'
            value_text = (f"directional reliability {reliability:.2f}  "
                          f"({res:.2f} \u00c5)")
        else:
            frac = mcolors.Normalize(vmin=vmin, vmax=vmax)(res)
            cmap_name = 'rainbow'
            value_text = f"resolution {res:.2f} \u00c5"

        rgba = mcm.get_cmap(cmap_name)(float(frac))
        color255 = [int(round(255 * c)) for c in rgba]
        try:
            for s_ in vol.surfaces:
                s_.vertex_colors = None   # uniform colour, not per-vertex
                s_.color = color255
        except Exception:
            return

        # The direction is the polar coordinate pair, as read off the DR plot;
        # the resolution is reported alongside it, not as the direction itself.
        text = (f"Direction: rot {azimuth:.0f}\u00b0, "
                f"tilt {zenith:.0f}\u00b0   |   {value_text}")
        self._dr_view_label.setText(text)
        self._set_view_readout(text)

    def _set_view_readout(self, text):
        """Live readout in the 3D view, just above the colour key. Created once
        then updated in place — going through the 2dlabels command every frame
        would fill the log."""
        from chimerax.label.label2d import find_label
        from chimerax.core.commands import run
        lbl = find_label(self.session, "fsc_dir_readout")
        if lbl is None:
            run(self.session,
                f'2dlabels create fsc_dir_readout text "{text}" '
                f'xpos 0.05 ypos 0.17 size 14')
        else:
            lbl.text = text
            lbl.update_drawing()

    def _clear_view_readout(self):
        from chimerax.label.label2d import find_label
        lbl = find_label(self.session, "fsc_dir_readout")
        if lbl is not None:
            try:
                lbl.delete()
            except Exception:
                pass

    def _refresh_dr_volume_list(self):
        from chimerax.map import Volume
        vols = self.session.models.list(type=Volume)

        for combo in (self._dr_half1_combo, self._dr_half2_combo):
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
        vol = self._dr_half1_combo.currentData()
        if vol is not None:
            try:
                step = float(vol.data.step[0])
                if step > 0:
                    self._dr_pix_spin.setValue(step)
            except Exception:
                pass

    # ---- compute ------------------------------------------------------

    def _on_dr_compute(self):
        from .directional_core import compute_directional

        half1 = self._dr_half1_combo.currentData()
        half2 = self._dr_half2_combo.currentData()
        mask = self._dr_mask_combo.currentData()
        if half1 is None or half2 is None:
            self.session.logger.error(
                "Directional Resolution: please select both half maps.")
            return
        if half1 is half2:
            msg = ("Half map 1 and half map 2 are the same model. FSC between a "
                   "map and itself is 1 at every frequency, so no direction ever "
                   "reaches the threshold and no resolution can be reported. "
                   "Select two different half maps.")
            self.session.logger.error(f"Directional Resolution: {msg}")
            self._dr_status_label.setText("Select two different half maps.")
            self._dr_status_label.setStyleSheet("color: #c0392b;")
            return
        v1 = half1.data.matrix()
        v2 = half2.data.matrix()
        if v1.shape != v2.shape:
            self.session.logger.error(
                f"Directional Resolution: the two half maps have different box "
                f"sizes \u2014 {half1.name} is {v1.shape} and {half2.name} is "
                f"{v2.shape}. They must be the same.")
            self._dr_status_label.setText("Half maps have different box sizes.")
            self._dr_status_label.setStyleSheet("color: #c0392b;")
            return
        if mask is not None and mask.data.matrix().shape != v1.shape:
            self.session.logger.error(
                f"Directional Resolution: the mask {mask.name} is "
                f"{mask.data.matrix().shape} but the half maps are {v1.shape}. "
                "Pick a mask on the same grid, or set Mask to None.")
            self._dr_status_label.setText("Mask does not match the map grid.")
            self._dr_status_label.setStyleSheet("color: #c0392b;")
            return

        self._dr_compute_btn.setEnabled(False)
        self._dr_status_label.setText("Computing\u2026")
        self._dr_status_label.setStyleSheet("")
        self._dr_progress.setValue(0)
        self._dr_progress.setVisible(True)
        self.session.logger.status("Computing Directional Resolution \u2026")

        v1 = v1.astype(np.float64, copy=True)
        v2 = v2.astype(np.float64, copy=True)
        mask_arr = (mask.data.matrix().astype(np.float64, copy=True)
                    if mask is not None else None)
        pixel_size = self._dr_pix_spin.value()
        cone_angle = self._dr_cone_spin.value()
        threshold = self._dr_thresh_spin.value()
        fast = self._dr_fast_check.isChecked()

        import threading, time

        def progress(frac):
            pct = int(max(0.0, min(1.0, frac)) * 100)
            self.session.ui.thread_safe(self._dr_set_progress, pct)

        def worker():
            t0 = time.perf_counter()
            try:
                out = compute_directional(
                    v1, v2, mask_arr,
                    sampling_rate=pixel_size, cone_angle=cone_angle,
                    threshold=threshold, fast=fast,
                    progress_callback=progress)
                elapsed = time.perf_counter() - t0
                self.session.ui.thread_safe(
                    self._dr_on_done, out, out["dr_grid"], elapsed)
            except Exception as e:
                self.session.ui.thread_safe(self._dr_on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _dr_set_progress(self, pct):
        self._dr_progress.setValue(pct)
        self._dr_progress.setFormat("Computing Directional Resolution…  %p%")

    def _dr_on_done(self, fso, dr, elapsed):
        if dr is None or not np.isfinite(dr).any():
            self._dr_on_error(
                "no direction reached the FSC threshold, so there is no "
                "resolution to report. This usually means the two half maps are "
                "identical or nearly so \u2014 check that you selected two "
                "genuinely different half maps, and not a map together with a "
                "copy or a mask.")
            return
        self._dr_fso = fso
        self._dr_dr = dr
        self._dr_draw_fso()
        self._dr_draw_polar()
        self._dr_view_check.setEnabled(True)
        self._dr_view_rel_check.setEnabled(True)
        self._dr_compute_btn.setEnabled(True)
        self._dr_progress.setVisible(False)
        self._dr_status_label.setText(f"Done in {elapsed:.1f} s")
        self._dr_status_label.setStyleSheet("color: #27ae60;")

        cr = fso["crossings"]
        lines = [f"Directional resolution (FSO) computed in {elapsed:.1f} s."]
        for thr in (0.9, 0.5, 0.1):
            if cr.get(thr):
                lines.append(f"  FSO = {thr}  \u2192  {cr[thr]:.2f} \u00c5")
        if cr.get(0.9) and cr.get(0.1):
            lines.append(f"  Anisotropy band: {cr[0.9]:.2f} \u2013 {cr[0.1]:.2f} \u00c5 "
                         f"(width {cr[0.9] - cr[0.1]:.2f} \u00c5)")
        lines.append(f"  Directional resolution range: "
                     f"{float(np.nanmin(dr)):.2f} \u2013 {float(np.nanmax(dr)):.2f} \u00c5")
        gfsc = fso.get("global_fsc")
        res = fso.get("res")
        if gfsc is not None and res is not None:
            below = np.nonzero(np.asarray(gfsc) <= self._dr_thresh_spin.value())[0]
            if below.size:
                lines.append(f"  Global FSC @ {self._dr_thresh_spin.value():.3f}: "
                             f"{float(res[below[0]]):.2f} \u00c5")
        lines.append(f"  Directions: {len(fso.get('directions', []))}")
        self.session.logger.info("\n".join(lines))
        self.session.logger.status("Directional resolution done.")
    def _dr_on_error(self, msg):
        self.session.logger.error(f"Directional resolution failed: {msg}")
        self._dr_status_label.setText(f"Error: {msg}")
        self._dr_status_label.setStyleSheet("color: #c0392b;")
        self._dr_compute_btn.setEnabled(True)
        self._dr_progress.setVisible(False)

    # ---- metric / histogram / color ----------------------------------

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

    def _select_combo_volume(self, combo, vol):
        """Point a dropdown at a specific volume (or at 'None' for a mask).
        Signals are blocked so this does not trigger the auto-fill handlers."""
        combo.blockSignals(True)
        idx = 0
        for i in range(combo.count()):
            if combo.itemData(i) is vol:
                idx = i
                break
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def set_fsc_inputs(self, map1, map2, mask=None, pixel_size=None,
                       threshold=None):
        """Make the Global FSC tab show what the `fsc` command actually used,
        rather than leaving the dropdowns on whatever was selected before."""
        self._refresh_volume_list()
        # Mode first: toggling it resets the threshold, so set that afterwards
        self._mode_map.setChecked(True)
        self._select_combo_volume(self._map1_combo, map1)
        self._select_combo_volume(self._map2_combo, map2)
        self._select_combo_volume(self._mask_combo, mask)
        if pixel_size:
            self._pix_spin.setValue(float(pixel_size))
        if threshold:
            self._thresh_spin.blockSignals(True)
            self._thresh_spin.setValue(float(threshold))
            self._thresh_spin.blockSignals(False)
        try:
            self._tabs.setCurrentIndex(0)      # bring the Global FSC tab forward
        except Exception:
            pass

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

        from matplotlib.ticker import FuncFormatter

        ax = self._ax
        ax.cla()
        self._fsc_click_marker = None
        self._fsc_last_idx = None
        self._fsc_curve = (freq_axis, np.asarray(fsc_values[1:]))
        # Same axis convention as the FSO plot: x is spatial frequency but the
        # ticks are labelled with the resolution they correspond to.
        ax.set_xlabel("Resolution (Å)", fontsize=9)
        ax.set_ylabel("FSC", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, x_max)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, p: f"{1.0 / v:.1f}" if v else ""))
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

        ax.legend(loc='upper right', fontsize=8)
        self._canvas.draw()

    # ==================================================================
    # Cleanup
    # ==================================================================

    def delete(self):
        if getattr(self, "_dr_view_handler", None) is not None:
            try:
                self._dr_view_check.setChecked(False)
                self._dr_view_rel_check.setChecked(False)
            except Exception:
                pass
        for h in self._handlers:
            self.session.triggers.remove_handler(h)
        super().delete()
