"""
gui.py - PyQt5 grafinė sąsaja

Akies dugno kraujagyslių analizės programa.
- Automatinis kraujagyslių matavimas ir klasifikavimas
- AVR (arterijų/venų santykio) skaičiavimas
- Akies pusės auto-detekcija pagal OD poziciją
- CSV saugojimo formatas atitinka
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel, QCheckBox,
    QProgressBar, QFileDialog, QScrollArea,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np

from processing import VesselProcessor


class ProcessingThread(QThread):
    """Apdorojimo gija """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path
        self.processor = None

    def run(self):
        try:
            self.processor = VesselProcessor()
            def on_progress(percent, message):
                self.progress.emit(percent, message)
            self.processor.set_progress_callback(on_progress)
            if not self.processor.load_image(self.image_path):
                self.finished.emit(None)
                return
            result = self.processor.run_full_processing()
            self.finished.emit(result)
        except Exception as e:
            print(f"Klaida apdorojant: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit(None)


class VesselsForm(QMainWindow):
    """Pagrindinė programos forma."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vessel Auto Measure - Python")
        self.setGeometry(100, 100, 1400, 900)

        self.processing_thread = None
        self.original_pixmap = None
        self.current_zoom = 1.0

        self.zoom_buttons = {}

        self._setup_ui()
        self._connect_signals()
        self._highlight_zoom_button(1.0)

    # ==================================================================
    # UI KŪRIMAS

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(350)
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)

        # Failų mygtukai
        self.cmdLoad = QPushButton("Atidaryti paveikslėlį")
        self.cmdSave = QPushButton("Išsaugoti rezultatus")
        self.cmdSave.setEnabled(False)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.cmdLoad)
        file_layout.addWidget(self.cmdSave)
        layout.addLayout(file_layout)

        # Mastelis
        zoom_box = QGroupBox("Mastelis")
        zoom_layout = QGridLayout(zoom_box)
        zoom_layout.setContentsMargins(5, 5, 5, 5)
        zoom_layout.setSpacing(2)
        self.vratio05 = QPushButton("1:2")
        self.vratio1 = QPushButton("1:1")
        self.vratio2 = QPushButton("2:1")
        self.vratio3 = QPushButton("3:1")
        zoom_layout.addWidget(self.vratio05, 0, 0)
        zoom_layout.addWidget(self.vratio1, 0, 1)
        zoom_layout.addWidget(self.vratio2, 1, 0)
        zoom_layout.addWidget(self.vratio3, 1, 1)
        layout.addWidget(zoom_box)
        self.zoom_buttons = {0.5: self.vratio05, 1.0: self.vratio1,
                             2.0: self.vratio2, 3.0: self.vratio3}

        # Atvaizdavimas
        display_box = QGroupBox("Atvaizdavimas")
        display_layout = QVBoxLayout(display_box)
        display_layout.setContentsMargins(5, 5, 5, 5)
        display_layout.setSpacing(1)
        self.preproc = QCheckBox("Apdorotas (CLAHE)")
        self.add_vessels = QCheckBox("Klasifikacija (A-raudona/V-mėlyna)")
        self.add_vessels_thn = QCheckBox("Suplonintos (mėlyna)")
        self.add_optic_disc = QCheckBox("OD + AVR zona")
        self.add_results = QCheckBox("Žymėjimai (tekstas)")
        self.preproc.setChecked(False)
        self.add_vessels.setChecked(True)
        self.add_vessels_thn.setChecked(False)
        self.add_optic_disc.setChecked(True)
        self.add_results.setChecked(False)
        display_layout.addWidget(self.preproc)
        display_layout.addWidget(self.add_vessels)
        display_layout.addWidget(self.add_vessels_thn)
        display_layout.addWidget(self.add_optic_disc)
        display_layout.addWidget(self.add_results)
        layout.addWidget(display_box)

        # Rezultatų lentelė
        results_box = QGroupBox("Rezultatai")
        results_layout = QVBoxLayout(results_box)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parametras", "Reikšmė"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)
        layout.addWidget(results_box, 1)

        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.loading_panel = QWidget()
        loading_layout = QVBoxLayout(self.loading_panel)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        self.progressBar = QProgressBar()
        self.label_status = QLabel("Atidarykite akies dugno paveikslėlį analizei.")
        self.label_status.setWordWrap(True)
        loading_layout.addWidget(self.progressBar)
        loading_layout.addWidget(self.label_status)
        layout.addWidget(self.loading_panel)
        self.loading_panel.setVisible(False)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.image_label)
        layout.addWidget(self.scrollArea)

        return panel

    # ==================================================================
    # SIGNALŲ PRIJUNGIMAS

    def _connect_signals(self):
        self.cmdLoad.clicked.connect(self.load_image_action)
        self.cmdSave.clicked.connect(self.save_results_action)

        self.vratio05.clicked.connect(lambda: self.set_zoom(0.5))
        self.vratio1.clicked.connect(lambda: self.set_zoom(1.0))
        self.vratio2.clicked.connect(lambda: self.set_zoom(2.0))
        self.vratio3.clicked.connect(lambda: self.set_zoom(3.0))

        for checkbox in [self.preproc, self.add_vessels, self.add_vessels_thn,
                         self.add_optic_disc, self.add_results]:
            checkbox.stateChanged.connect(self.update_display)

    # ==================================================================
    # ZOOM PARYŠKINIMAS

    def _highlight_zoom_button(self, active_ratio: float):
        """Bold fontas aktyviam mastelio mygtukui."""
        for ratio, btn in self.zoom_buttons.items():
            font = btn.font()
            font.setBold(ratio == active_ratio)
            btn.setFont(font)

    # ==================================================================
    # FAILŲ OPERACIJOS

    def load_image_action(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pasirinkite paveikslėlį", "",
            "Paveikslėliai (*.jpeg *.jpg *.png *.tif *.tiff *.ppm *.bmp)"
        )
        if not file_path:
            return

        self.image_label.clear()
        self.loading_panel.setVisible(True)
        self.cmdSave.setEnabled(False)
        self.progressBar.setValue(0)
        self._update_results_table(None)

        self.processing_thread = ProcessingThread(file_path)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def save_results_action(self):
        """Rezultatų išsaugojimas"""
        if self.processing_thread and self.processing_thread.processor:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Išsaugoti rezultatą", "",
                "PNG (*.png);;JPEG (*.jpg)"
            )
            if file_path:
                import os
                base_path = os.path.splitext(file_path)[0]
                self.processing_thread.processor.save_results(base_path)
                self.label_status.setText(
                    f"Išsaugota: {base_path}_rez.png + _rez.csv + _rez2.csv"
                )

    # ==================================================================
    # PROGRESO IR APDOROJIMO PABAIGA

    def update_progress(self, value: int, message: str):
        self.progressBar.setValue(value)
        self.label_status.setText(f"Apdorojama... {message}")

    def on_processing_finished(self, result):
        self.loading_panel.setVisible(False)

        if result is None or not result.success:
            error_msg = "Nepavyko apdoroti paveikslėlio."
            if result and result.error_message:
                error_msg = result.error_message
            self.label_status.setText(f"Klaida: {error_msg}")
            self._update_results_table(None)
            return

        self.cmdSave.setEnabled(True)

        status_text = "Apdorojimas baigtas."
        if result.od_r > 0:
            status_text += f" OD: ({result.od_x}, {result.od_y})"
        if result.avr:
            avr_total = result.avr.get('total', 0)
            if avr_total > 0:
                status_text += f" AVR: {avr_total:.3f}"
        self.label_status.setText(status_text)

        self._update_results_table(result)
        self.update_display()

    # ==================================================================
    # MASTELIS IR ATVAIZDAVIMAS

    def set_zoom(self, factor: float):
        self.current_zoom = factor
        self._highlight_zoom_button(factor)
        self.update_display()

    def update_display(self):
        if not (self.processing_thread and self.processing_thread.processor):
            return

        processor = self.processing_thread.processor
        options = {
            'show_preprocessed': self.preproc.isChecked(),
            'show_vessels': self.add_vessels.isChecked(),
            'show_vessels_thn': self.add_vessels_thn.isChecked(),
            'show_optic_disc': self.add_optic_disc.isChecked(),
            'show_labels': self.add_results.isChecked(),
        }

        display_img = processor.get_result_image(options)
        if display_img is not None:
            self._display_cv_image(display_img)

    def _display_cv_image(self, cv_img: np.ndarray):
        h, w = cv_img.shape[:2]
        ch = cv_img.shape[2] if len(cv_img.shape) == 3 else 1
        if ch == 1:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            ch = 3
        bytes_per_line = ch * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.original_pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = self.original_pixmap.scaled(
            int(w * self.current_zoom), int(h * self.current_zoom),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    # ==================================================================
    # REZULTATŲ LENTELĖ

    def _update_results_table(self, result):
        self.results_table.setRowCount(0)
        if result is None:
            return

        rows = []

        # Baziniai
        rows.append(("▶ BAZINIAI", ""))
        rows.append(("Skalė (sc)", f"{result.scale:.4f}"))
        if hasattr(result, 'img_width') and result.img_width:
            rows.append(("Dydis", f"{result.img_width}x{result.img_height}"))

        # Optinis diskas
        rows.append(("▶ OPTINIS DISKAS", ""))
        if result.od_r > 0:
            rows.append(("Centras", f"({result.od_x}, {result.od_y})"))
            rows.append(("Spindulys", f"{result.od_r} px"))
            rows.append(("Skersmuo", f"{result.od_r * 2} px"))
        else:
            rows.append(("Būsena", "Nerastas"))

        # Kraujagyslės
        rows.append(("▶ KRAUJAGYSLĖS", ""))
        if hasattr(result, 'measurements') and result.measurements:
            rows.append(("Matavimai", str(len(result.measurements))))
        else:
            rows.append(("Matavimai", "0"))

        if hasattr(result, 'stats') and result.stats:
            stats = result.stats
            means = stats.get('mean', [0, 0, 0, 0])
            sds = stats.get('sd', [0, 0, 0, 0])
            counts = stats.get('count', [0, 0, 0, 0])
            if counts[0] > 0:
                rows.append(("A↑ (arterija)", f"{means[0]:.2f}±{sds[0]:.2f} (n={counts[0]})"))
            if counts[1] > 0:
                rows.append(("V↑ (vena)", f"{means[1]:.2f}±{sds[1]:.2f} (n={counts[1]})"))
            if counts[2] > 0:
                rows.append(("A↓ (arterija)", f"{means[2]:.2f}±{sds[2]:.2f} (n={counts[2]})"))
            if counts[3] > 0:
                rows.append(("V↓ (vena)", f"{means[3]:.2f}±{sds[3]:.2f} (n={counts[3]})"))
            art_sum = means[0] * counts[0] + means[2] * counts[2]
            art_count = counts[0] + counts[2]
            ven_sum = means[1] * counts[1] + means[3] * counts[3]
            ven_count = counts[1] + counts[3]
            if art_count > 0:
                rows.append(("Arterijos (vid.)", f"{art_sum / art_count:.2f} px"))
            if ven_count > 0:
                rows.append(("Venos (vid.)", f"{ven_sum / ven_count:.2f} px"))

        # AVR
        rows.append(("▶ AVR RODIKLIAI", ""))
        if hasattr(result, 'avr') and result.avr:
            avr_top = result.avr.get('top', 0)
            avr_btm = result.avr.get('bottom', 0)
            avr_total = result.avr.get('total', 0)
            if avr_top > 0:
                rows.append(("Viršutinis", f"{avr_top:.4f}"))
            if avr_btm > 0:
                rows.append(("Apatinis", f"{avr_btm:.4f}"))
            if avr_total > 0:
                rows.append(("Bendras", f"{avr_total:.4f}"))
                if avr_total < 0.67:
                    interp = "Žemas"
                elif avr_total > 0.9:
                    interp = "Aukštas"
                else:
                    interp = "Normalus"
                rows.append(("Interpretacija", interp))
        else:
            rows.append(("Būsena", "Neskaičiuota"))

        # Trukmė
        if hasattr(result, 'timing') and result.timing:
            rows.append(("▶ TRUKMĖ", ""))
            timing = result.timing
            if 'mask' in timing:
                rows.append(("Kaukė", f"{timing['mask']:.2f} s"))
            if 'preprocess' in timing:
                rows.append(("Preprocessing", f"{timing['preprocess']:.2f} s"))
            if 'extraction' in timing:
                rows.append(("Išskyrimas", f"{timing['extraction']:.2f} s"))
            if 'optic_disc' in timing:
                rows.append(("Optinis diskas", f"{timing['optic_disc']:.2f} s"))
            if 'measurement' in timing:
                rows.append(("Matavimai", f"{timing['measurement']:.2f} s"))
            if 'total' in timing:
                rows.append(("VISO", f"{timing['total']:.2f} s"))

        # Užpildyti lentelę
        self.results_table.setRowCount(len(rows))
        for i, (param, value) in enumerate(rows):
            param_item = QTableWidgetItem(param)
            value_item = QTableWidgetItem(value)
            if param.startswith("▶") or param == "VISO":
                font = param_item.font()
                font.setBold(True)
                param_item.setFont(font)
                if param == "VISO":
                    value_item.setFont(font)
            self.results_table.setItem(i, 0, param_item)
            self.results_table.setItem(i, 1, value_item)