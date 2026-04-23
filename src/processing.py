"""
processing.py - Pagrindinis apdorojimo orchestratorius

Apjungia visus modulius ir valdo pilną apdorojimo pipeline:
1. Kaukės sukūrimas (masking)
2. Paveikslėlio paruošimas (preprocessing)
3. Kraujagyslių išskyrimas (vessel_extraction)
4. Optinio disko aptikimas (optic_disc)
5. Matavimai (vessel_measurement)
6. Klasifikacija ir AVR skaičiavimas
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

# Modulių importai
from masking import create_fundus_mask, normingMask
from preprocessing import preprocessing1, preprocessing2, preprocessing3, preprocessing4
from vessel_extraction import bwe1, thinning, cleareIsolated
from optic_disc import detect_optic_disc, opticDiscInit1, opticDisc2
from vessel_measurement import (
    measure_od_points, classifyVessels, calculate_avr,
    measureVesselPoint, getProfilePoints, getProfileData
)
from config import Config, ImageStore, PI


@dataclass
class ProcessingResult:
    """Apdorojimo rezultatų konteineris."""
    success: bool = False
    error_message: str = ""

    # Optinio disko parametrai
    od_x: int = 0
    od_y: int = 0
    od_r: int = 0

    # Skalė
    scale: float = 1.0

    # Paveikslėlio dydis
    img_width: int = 0
    img_height: int = 0

    # Matavimai
    measurements: list = field(default_factory=list)

    # Klasifikacija
    classification: dict = field(default_factory=dict)

    # AVR
    avr: dict = field(default_factory=dict)

    # Statistika
    stats: dict = field(default_factory=dict)

    # Skaičiavimų trukmė (sekundėmis)
    timing: dict = field(default_factory=dict)


class VesselProcessor:
    """
    Pagrindinis kraujagyslių analizės procesorius.
    """

    def __init__(self, image_path: Optional[str] = None):
        """
        Inicializuoja procesorių.

        Args:
            image_path: Kelias iki paveikslėlio (nebūtinas)
        """
        self.config = Config()
        self.images = ImageStore()
        self.result = ProcessingResult()

        # Progress callback
        self._progress_callback: Optional[Callable[[int, str], None]] = None

        if image_path:
            self.load_image(image_path)

    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Nustato progreso callback funkciją."""
        self._progress_callback = callback

    def _report_progress(self, percent: int, message: str):
        """Praneša apie progresą."""
        if self._progress_callback:
            self._progress_callback(percent, message)

    def load_image(self, image_path: str) -> bool:
        """
        Įkelia paveikslėlį.

        Args:
            image_path: Kelias iki paveikslėlio

        Returns:
            True jei pavyko
        """
        if not os.path.exists(image_path):
            self.result.error_message = f"Failas nerastas: {image_path}"
            return False

        self.images.image = cv2.imread(image_path)

        if self.images.image is None:
            self.result.error_message = f"Nepavyko įkelti: {image_path}"
            return False

        # Patikrinti ar paveikslėlis yra grayscale ir konvertuoti į RGB
        if len(self.images.image.shape) == 2:
            # Grayscale -> RGB
            self.images.image = cv2.cvtColor(self.images.image, cv2.COLOR_GRAY2BGR)
        elif self.images.image.shape[2] == 1:
            # Single channel -> RGB
            self.images.image = cv2.cvtColor(self.images.image, cv2.COLOR_GRAY2BGR)

        # Išsaugoti originalų paveikslėlį
        self.images.img_orig = self.images.image.copy()

        # Išsaugoti matmenis (skalė bus nustatyta create_mask metu)
        height, width = self.images.image.shape[:2]
        self.result.img_width = width
        self.result.img_height = height

        self._report_progress(5, "Paveikslėlis įkeltas")
        return True

    def create_mask(self) -> bool:
        """
        Sukuria akies dugno kaukę.

        Returns:
            True jei pavyko
        """
        if self.images.image is None:
            self.result.error_message = "Paveikslėlis neįkeltas"
            return False

        self._report_progress(10, "Kuriama kaukė...")

        # Išskirti žalią kanalą
        channels = cv2.split(self.images.image)
        self.images.img_g = channels[1]  # Žalias kanalas

        # Sukurti kaukę
        self.images.img_mask, sc = create_fundus_mask(self.images.img_g.copy())

        # Nustatyti skalę pagal kaukę (C++ tvarka: createMask → setScale)
        self.config.setScale(self.images.img_mask)
        self.result.scale = self.config.scale.sc

        self._report_progress(15, "Kaukė sukurta")
        return True

    def preprocess(self) -> bool:
        """
        Atlieka paveikslėlio paruošimą.

        Returns:
            True jei pavyko
        """
        if self.images.img_mask is None:
            self.result.error_message = "Kaukė nesukurta"
            return False

        self._report_progress(20, "Ruošiamas paveikslėlis...")

        sc = self.config.scale.sc

        # Preprocessing1 - kraujagyslėms
        self.images.img_prep1 = preprocessing1(
            self.images.image.copy(),
            self.images.img_mask,
            sc
        )

        self._report_progress(25, "Preprocessing 1 baigtas")

        # Preprocessing2 - optiniam diskui
        self.images.img_prep2 = preprocessing2(self.images.image.copy())

        self._report_progress(30, "Preprocessing 2 baigtas")

        # Preprocessing3 - matavimams
        self.images.img_prep3 = preprocessing3(
            self.images.image.copy(),
            self.images.img_mask
        )

        self._report_progress(35, "Preprocessing 3 baigtas")

        # Preprocessing4 - matavimams išplėstinis
        self.images.img_prep4 = preprocessing4(
            self.images.image.copy(),
            self.images.img_mask
        )

        self._report_progress(40, "Preprocessing baigtas")
        return True

    def extract_vessels(self) -> bool:
        """
        Išskiria kraujagysles.

        Returns:
            True jei pavyko
        """
        if self.images.img_prep1 is None:
            self.result.error_message = "Preprocessing neatliktas"
            return False

        self._report_progress(45, "Išskiriamos kraujagyslės...")

        sc = self.config.scale.sc

        # Išskirti žalią kanalą iš preprocessing1
        if len(self.images.img_prep1.shape) == 3:
            channels = cv2.split(self.images.img_prep1)
            img_green = channels[1]
        else:
            img_green = self.images.img_prep1

        # BWE1 - kraujagyslių išskyrimas
        self.images.img_vsl, self.images.img_vsl_thn = bwe1(
            img_green.copy(),
            self.images.img_mask,
            sc
        )

        self._report_progress(55, "Kraujagyslės išskirtos")
        return True

    def detect_optic_disc(self) -> bool:
        """
        Aptinka optinį diską.

        Returns:
            True jei pavyko
        """
        if self.images.img_vsl_thn is None:
            self.result.error_message = "Kraujagyslės neišskirtos"
            return False

        self._report_progress(60, "Ieškomas optinis diskas...")

        sc = self.config.scale.sc

        #  img_g = (ch[0] + ch[1] + ch[2]) / 3;
        if len(self.images.img_prep2.shape) == 3:
            channels = cv2.split(self.images.img_prep2)
            # Visų kanalų vidurkis (kaip C++ kode)
            img_green = (channels[0].astype(np.float32) +
                         channels[1].astype(np.float32) +
                         channels[2].astype(np.float32)) / 3.0
            img_green = img_green.astype(np.uint8)
        else:
            img_green = self.images.img_prep2

        od_x, od_y, od_r = detect_optic_disc(
            img_green,
            self.images.img_vsl_thn,
            self.images.img_mask,
            sc
        )

        # Išsaugoti rezultatus
        self.config.optic_disc.x = od_x
        self.config.optic_disc.y = od_y
        self.config.optic_disc.r = od_r

        self.result.od_x = od_x
        self.result.od_y = od_y
        self.result.od_r = od_r

        if od_x == 0 and od_y == 0:
            self._report_progress(65, "Optinis diskas nerastas")
            return False

        self._report_progress(70, f"Optinis diskas rastas: ({od_x}, {od_y}), r={od_r}")
        return True

    def measure_vessels(self) -> bool:
        """
        Atlieka kraujagyslių matavimus.

        Returns:
            True jei pavyko
        """
        if self.result.od_x == 0:
            self.result.error_message = "Optinis diskas neaptiktas"
            return False

        self._report_progress(75, "Atliekami matavimai...")

        sc = self.config.scale.sc
        od_x = self.result.od_x
        od_y = self.result.od_y
        od_r = self.result.od_r

        # Išskirti kanalus iš preprocessing3
        if len(self.images.img_prep3.shape) == 3:
            channels = cv2.split(self.images.img_prep3)
            img_g = channels[1]  # Žalias
            img_r = channels[2]  # Raudonas
        else:
            img_g = self.images.img_prep3
            img_r = self.images.img_prep3


        b3 = int(3 * sc)
        if b3 % 2 == 0:
            b3 += 1
        b3 = max(3, b3)  # Minimalus dydis 3
        B3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b3, b3))
        # Opening operacija (erode + dilate)
        img_g2 = cv2.erode(img_g, B3)
        img_g2 = cv2.dilate(img_g2, B3)

        # Matavimai aplink optinį diską
        measurements = measure_od_points(
            od_x, od_y, od_r,
            img_g, img_g2, img_r,
            self.images.img_mask,
            self.images.img_vsl_thn,
            sc,
            odr_mult=self.config.vessel.odr_mult if hasattr(self.config, 'vessel') else [2.0, 2.5]
        )

        self.result.measurements = measurements

        self._report_progress(85, f"Atlikta {len(measurements)} matavimų")
        return len(measurements) > 0

    def classify_vessels(self) -> bool:
        """
        Klasifikuoja kraujagysles į arterijas ir venas.

        Returns:
            True jei pavyko
        """
        if not self.result.measurements:
            self.result.error_message = "Nėra matavimų"
            return False

        self._report_progress(90, "Klasifikuojamos kraujagyslės...")

        # Konvertuoti matavimus į vslsave formatą
        vslsave = []
        for m in self.result.measurements:
            # [type, x, y, rad, width, features...]
            row = [0, m['x'], m['y'], m['angle'], m['width']] + m['features']
            vslsave.append(row)

        # Klasifikuoti
        classification = classifyVessels(
            vslsave,
            self.result.od_x,
            self.result.od_y,
            norm_method='spatial_noself',
            kmeans_method='kmeans'
        )

        self.result.classification = classification

        # Priskirti stats (net jei tuščias)
        self.result.stats = classification.get('stats', {})

        # Apskaičiuoti AVR jei yra stats su count > 0
        if self.result.stats and any(c > 0 for c in self.result.stats.get('count', [])):
            self.result.avr = calculate_avr(self.result.stats)

        self._report_progress(95, "Klasifikacija baigta")
        return True

    def run_full_processing(self) -> ProcessingResult:
        """
        Vykdo pilną apdorojimo pipeline.

        Returns:
            ProcessingResult su visais rezultatais
        """
        import time

        timing = {}
        total_start = time.time()

        self._report_progress(0, "Pradedamas apdorojimas...")

        # 1. Kaukės sukūrimas
        step_start = time.time()
        if not self.create_mask():
            self.result.success = False
            return self.result
        timing['mask'] = time.time() - step_start

        # 2. Preprocessing
        step_start = time.time()
        if not self.preprocess():
            self.result.success = False
            return self.result
        timing['preprocess'] = time.time() - step_start

        # 3. Kraujagyslių išskyrimas
        step_start = time.time()
        if not self.extract_vessels():
            self.result.success = False
            return self.result
        timing['extraction'] = time.time() - step_start

        # 4. Optinio disko aptikimas
        step_start = time.time()
        if not self.detect_optic_disc():
            # Tęsiame net jei OD nerastas
            pass
        timing['optic_disc'] = time.time() - step_start

        # 5. Matavimai
        step_start = time.time()
        if self.result.od_x > 0:
            if not self.measure_vessels():
                pass  # Tęsiame net jei matavimai nepavyko

            # 6. Klasifikacija
            if self.result.measurements:
                self.classify_vessels()
        timing['measurement'] = time.time() - step_start

        # Bendra trukmė
        timing['total'] = time.time() - total_start
        self.result.timing = timing

        self._report_progress(100, "Apdorojimas baigtas")
        self.result.success = True
        return self.result

    def get_result_image(self, options: dict = None) -> np.ndarray:
        """
        Sukuria rezultatų paveikslėlį

        Vizualizacija:
        - OD centras, riba, 1.5×rOD ir 3×rOD zonos
        - Didžiausios kraujagyslės pažymėtos su užrašais

        Args:
            options: Atvaizdavimo parinktys

        Returns:
            Rezultatų paveikslėlis
        """
        import math

        if options is None:
            options = {
                'show_preprocessed': False,
                'show_vessels': True,
                'show_vessels_thn': False,
                'show_optic_disc': True,
                'show_measurements': True,
                'show_labels': False  # Tekstas išjungtas pagal nutylėjimą
            }

        # Pasirinkti bazinį paveikslėlį
        prep_img = self.images.img_prep4 if self.images.img_prep4 is not None else self.images.img_prep1
        if options.get('show_preprocessed') and prep_img is not None:
            if len(prep_img.shape) == 3:
                result = prep_img.copy()
            elif len(prep_img.shape) == 2:
                result = cv2.cvtColor(prep_img, cv2.COLOR_GRAY2BGR)
            else:
                result = prep_img.copy()
        else:
            if self.images.img_orig is not None:
                result = self.images.img_orig.copy()
            elif self.images.image is not None:
                result = self.images.image.copy()
            else:
                return np.zeros((100, 100, 3), dtype=np.uint8)

        # =====================================================================
        # SUPLONINTOS KRAUJAGYSLĖS (jei įjungta)
        if options.get('show_vessels_thn') and self.images.img_vsl_thn is not None:
            thn_mask = self.images.img_vsl_thn > 0
            result[thn_mask] = [255, 200, 0]  # Šviesiai mėlyna

        # =====================================================================
        # KRAUJAGYSLĖS SU SPALVOTAIS SEGMENTAIS
        # Tik 4 pagrindinės (plačiausios) kraujagyslės.
        # Kontūrai apkarpomi iki 1.5–3.0 rOD zonos.
        # Matavimo taškai žymimi tamsesne spalva.
        # Arterijos = RAUDONA, Venos = MĖLYNA

        artery_tm, artery_bm = -1, -1
        vein_tm, vein_bm = -1, -1
        max_art_top, max_art_btm = 0, 0
        max_ven_top, max_ven_btm = 0, 0

        if options.get('show_vessels') and self.result.measurements and self.result.classification:
            classes = self.result.classification.get('classes', [])
            od_x = self.result.od_x if self.result.od_x > 0 else result.shape[1] // 2
            od_y = self.result.od_y if self.result.od_y > 0 else result.shape[0] // 2
            od_r = self.result.od_r if self.result.od_r > 0 else 100

            # Zonos ribos
            r_inner = 1.5 * od_r
            r_outer = 3.0 * od_r
            r_inner_sq = r_inner * r_inner
            r_outer_sq = r_outer * r_outer

            # 1. Surasti didžiausias kraujagysles kiekvienoje kategorijoje
            for i, m in enumerate(self.result.measurements):
                cls = classes[i] if i < len(classes) else 0
                width = m.get('width', 0)
                if cls == 1 and width > max_art_top:
                    max_art_top = width; artery_tm = i
                elif cls == 2 and width > max_ven_top:
                    max_ven_top = width; vein_tm = i
                elif cls == 3 and width > max_art_btm:
                    max_art_btm = width; artery_bm = i
                elif cls == 4 and width > max_ven_btm:
                    max_ven_btm = width; vein_bm = i

            # Spalvos (BGR)
            art_color = (0, 0, 255)     # ryški raudona
            art_meas  = (0, 0, 120)     # tamsi raudona (matavimo taškai)
            ven_color = (255, 0, 0)     # ryški mėlyna
            ven_meas  = (120, 0, 0)     # tamsi mėlyna (matavimo taškai)

            PI = math.pi

            # 2. Piešti TIK 4 pagrindines kraujagysles, apkarpytas iki zonos
            for idx, fill_color, meas_color in [
                (artery_tm, art_color, art_meas),
                (artery_bm, art_color, art_meas),
                (vein_tm, ven_color, ven_meas),
                (vein_bm, ven_color, ven_meas),
            ]:
                if idx == -1 or idx >= len(self.result.measurements):
                    continue

                m = self.result.measurements[idx]
                vp = m.get('vessel_points')
                if vp is None:
                    continue

                v_rad = vp['v_rad']
                v_len = vp['v_len']
                v_cx = vp['v_cx']
                v_cy = vp['v_cy']

                # Filtruoti taškus pagal zoną (1.5–3.0 rOD)
                in_zone = []
                for j in range(len(v_len)):
                    dx = v_cx[j] - od_x
                    dy = v_cy[j] - od_y
                    dist_sq = dx * dx + dy * dy
                    in_zone.append(r_inner_sq <= dist_sq <= r_outer_sq)

                # Kontūrai tik tarp gretimų taškų, kurie ABU yra zonoje
                contours = []
                for j in range(1, len(v_len)):
                    if not (in_zone[j] and in_zone[j-1]):
                        continue

                    pf_x = int(v_cx[j] + v_len[j] * 0.5 * math.cos(v_rad[j]))
                    pf_y = int(v_cy[j] + v_len[j] * 0.5 * math.sin(v_rad[j]))
                    pt_x = int(v_cx[j] + v_len[j] * 0.5 * math.cos(v_rad[j] + PI))
                    pt_y = int(v_cy[j] + v_len[j] * 0.5 * math.sin(v_rad[j] + PI))

                    pfl_x = int(v_cx[j-1] + v_len[j-1] * 0.5 * math.cos(v_rad[j-1]))
                    pfl_y = int(v_cy[j-1] + v_len[j-1] * 0.5 * math.sin(v_rad[j-1]))
                    ptl_x = int(v_cx[j-1] + v_len[j-1] * 0.5 * math.cos(v_rad[j-1] + PI))
                    ptl_y = int(v_cy[j-1] + v_len[j-1] * 0.5 * math.sin(v_rad[j-1] + PI))

                    contour = np.array([
                        [pf_x, pf_y], [pt_x, pt_y],
                        [ptl_x, ptl_y], [pfl_x, pfl_y]
                    ], dtype=np.int32)
                    contours.append(contour)

                if contours:
                    cv2.drawContours(result, contours, -1, fill_color, -1)

                # Matavimo taškai — tamsesni skerspjūvio brūkšneliai (tik zonoje)
                zone_indices = [j for j in range(len(v_len)) if in_zone[j]]
                show_labels = options.get('show_labels', False)
                prev_lx, prev_ly = -999, -999  # ankstesnės etiketės pozicija

                for zi, j in enumerate(zone_indices):
                    pf_x = int(v_cx[j] + v_len[j] * 0.5 * math.cos(v_rad[j]))
                    pf_y = int(v_cy[j] + v_len[j] * 0.5 * math.sin(v_rad[j]))
                    pt_x = int(v_cx[j] + v_len[j] * 0.5 * math.cos(v_rad[j] + PI))
                    pt_y = int(v_cy[j] + v_len[j] * 0.5 * math.sin(v_rad[j] + PI))
                    cv2.line(result, (pf_x, pf_y), (pt_x, pt_y), meas_color, 1)

                    # Pločio etiketė kas 3-čią atkarpą (tik su Žymėjimai)
                    if show_labels and zi % 3 == 0:
                        # Parinkti tick galą toliau nuo OD (kad neužšoktų ant segmento)
                        d_pf = (pf_x - od_x)**2 + (pf_y - od_y)**2
                        d_pt = (pt_x - od_x)**2 + (pt_y - od_y)**2
                        if d_pf >= d_pt:
                            ex, ey = pf_x, pf_y
                            nx = pf_x - int(v_cx[j])
                            ny = pf_y - int(v_cy[j])
                        else:
                            ex, ey = pt_x, pt_y
                            nx = pt_x - int(v_cx[j])
                            ny = pt_y - int(v_cy[j])
                        # Pastumti 5px toliau nuo kraujagyslės centro
                        nd = math.sqrt(nx*nx + ny*ny) if (nx*nx + ny*ny) > 0 else 1
                        lx = ex + int(5 * nx / nd)
                        ly = ey + int(5 * ny / nd)

                        # Praleisti jei per arti ankstesnės etiketės
                        if (lx - prev_lx)**2 + (ly - prev_ly)**2 >= 20*20:
                            cv2.putText(result, f"{v_len[j]:.1f}",
                                       (lx, ly),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, meas_color, 1)
                            prev_lx, prev_ly = lx, ly

        # =====================================================================
        # OPTINIS DISKAS IR AVR ZONOS
        if options.get('show_optic_disc') and self.result.od_r > 0:
            center = (self.result.od_x, self.result.od_y)
            od_r = self.result.od_r

            # Centras (baltas taškas)
            cv2.circle(result, center, 5, (255, 255, 255), -1)

            # OD riba (balta)
            cv2.circle(result, center, od_r, (255, 255, 255), 2)

            # 1.5 × rOD - AVR matavimų zonos pradžia (balta)
            cv2.circle(result, center, int(1.5 * od_r), (255, 255, 255), 1)

            # 3.0 × rOD - AVR matavimų zonos pabaiga (balta)
            cv2.circle(result, center, int(3.0 * od_r), (255, 255, 255), 1)

        # =====================================================================
        # TEKSTO BLOKAS
        # Baltas stačiakampis viršutiniame kairiajame kampe su matavimais
        if options.get('show_labels', False) and self.result.stats:
            stats = self.result.stats
            means = stats.get('mean', [0, 0, 0, 0])
            avr = self.result.avr or {}
            avr_top = avr.get('top', 0)
            avr_btm = avr.get('bottom', 0)

            artt = f"Artery = {means[0]:.2f}px" if means[0] != 0 else "Artery = -"
            vnt = f"Vein = {means[1]:.2f}px" if means[1] != 0 else "Vein = -"
            rtt = f"AVR = {avr_top:.4f}" if avr_top > 0 else "AVR = -"
            artb = f"Artery = {means[2]:.2f}px" if means[2] != 0 else "Artery = -"
            vnb = f"Vein = {means[3]:.2f}px" if means[3] != 0 else "Vein = -"
            rtb = f"AVR = {avr_btm:.4f}" if avr_btm > 0 else "AVR = -"

            cv2.rectangle(result, (1, 1), (400, 170), (255, 255, 255), -1)

            font = cv2.FONT_HERSHEY_TRIPLEX  # C++ font=4
            cv2.putText(result, "Top measurements", (5, 20), font, 0.5, (0, 0, 0), 1)
            cv2.putText(result, artt, (5, 40), font, 0.4, (0, 0, 255), 1)  # Raudona = arterija
            cv2.putText(result, vnt, (5, 60), font, 0.4, (255, 0, 0), 1)   # Mėlyna = vena
            cv2.putText(result, rtt, (5, 80), font, 0.4, (0, 0, 0), 1)

            cv2.putText(result, "Bottom measurements", (5, 100), font, 0.5, (0, 0, 0), 1)
            cv2.putText(result, artb, (5, 120), font, 0.4, (0, 0, 255), 1)
            cv2.putText(result, vnb, (5, 140), font, 0.4, (255, 0, 0), 1)
            cv2.putText(result, rtb, (5, 160), font, 0.4, (0, 0, 0), 1)

        # =====================================================================
        # OD IR KRAUJAGYSLIŲ ŽYMĖJIMAI ANT PAVEIKSLIUKO
        if options.get('show_labels', False) and self.result.od_r > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            center = (self.result.od_x, self.result.od_y)
            od_r = self.result.od_r

            cv2.putText(result, f"OD ({self.result.od_x},{self.result.od_y})",
                       (center[0] - 30, center[1] - od_r - 10), font, 0.4, (255, 255, 255), 1)
            cv2.putText(result, "rOD", (center[0] + od_r + 5, center[1] - 15),
                       font, 0.4, (255, 255, 255), 1)
            cv2.putText(result, "1.5rOD", (center[0] + int(1.5 * od_r) + 5, center[1] + 5),
                       font, 0.4, (255, 255, 255), 1)
            cv2.putText(result, "3rOD", (center[0] + int(3.0 * od_r) + 5, center[1] + 25),
                       font, 0.4, (255, 255, 255), 1)

        # =====================================================================
        # DIDŽIAUSIŲ KRAUJAGYSLIŲ ŽYMĖJIMAI
        if options.get('show_labels', False) and self.result.measurements and self.result.classification:
            font = cv2.FONT_HERSHEY_SIMPLEX

            od_x = self.result.od_x if self.result.od_x > 0 else result.shape[1] // 2
            od_y = self.result.od_y if self.result.od_y > 0 else result.shape[0] // 2

            for idx, label, color in [
                (artery_tm, "Virsutine A", (0, 0, 255)),
                (vein_tm, "Virsutine V", (255, 0, 0)),
                (artery_bm, "Apatine A", (0, 0, 255)),
                (vein_bm, "Apatine V", (255, 0, 0))
            ]:
                if idx >= 0:
                    m = self.result.measurements[idx]
                    x, y = int(m['x']), int(m['y'])
                    dx = x - od_x
                    label_x = x + (40 if dx > 0 else -100)
                    label_y = y - 5 if "Virs" in label else y + 15
                    cv2.putText(result, label, (label_x, label_y), font, 0.35, color, 1)

        return result

    def save_results(self, output_path: str) -> bool:
        """
        Išsaugo rezultatus: paveikslėlį + CSV.

        """
        try:
            # Rezultatų paveikslėlis (C++: imwrite(rawname + "_rez." + ext, image_rez))
            result_image = self.get_result_image()
            cv2.imwrite(f"{output_path}_rez.png", result_image)

            # CSV #1: _rez.csv (C++ save_text — pagrindiniai rezultatai)
            self._save_csv_rez(f"{output_path}_rez.csv")

            # CSV #2: _rez2.csv (C++ save_text2 — detalūs duomenys)
            self._save_csv_rez2(f"{output_path}_rez2.csv")

            return True
        except Exception as e:
            self.result.error_message = str(e)
            return False

    def _save_csv_rez(self, csv_path: str):
        """
        Išsaugo _rez.csv
        File;Artery top;;;Vein top;;;Ratio top;Artery bottom;;;Vein bottom;;;Ratio bottom;
        ;length;x;y;length;x;y;;length;x;y;length;x;y;;
        """
        import os

        with open(csv_path, 'w') as f:

            f.write("File;Artery top;;;Vein top;;;Ratio top;Artery bottom;;;Vein bottom;;;Ratio bottom;\n")
            f.write(";length;x;y;length;x;y;;length;x;y;length;x;y;;\n")

            # Duomenų eilutė
            filename = os.path.basename(csv_path).replace("_rez.csv", "")

            classification = self.result.classification or {}
            classes = classification.get('classes', [])
            stats = self.result.stats or {}
            means = stats.get('mean', [0, 0, 0, 0])

            # Rasti didžiausias kraujagysles (kaip classifyVessels)
            artery_tm, vein_tm, artery_bm, vein_bm = -1, -1, -1, -1
            max_w = [0, 0, 0, 0]

            for i, m in enumerate(self.result.measurements):
                cls = classes[i] if i < len(classes) else 0
                w = m.get('width', 0)
                if cls == 1 and w > max_w[0]:
                    max_w[0] = w; artery_tm = i
                elif cls == 2 and w > max_w[1]:
                    max_w[1] = w; vein_tm = i
                elif cls == 3 and w > max_w[2]:
                    max_w[2] = w; artery_bm = i
                elif cls == 4 and w > max_w[3]:
                    max_w[3] = w; vein_bm = i

            def fmt(idx):
                """Formatuoja matavimą: width;x;y"""
                if idx >= 0 and idx < len(self.result.measurements):
                    m = self.result.measurements[idx]
                    return f"{m['width']:.4f};{m['x']:.1f};{m['y']:.1f}"
                return "-;-;-"

            avr = self.result.avr or {}
            top_ratio = avr.get('top', 0) if avr.get('top', 0) > 0 else -1
            btm_ratio = avr.get('bottom', 0) if avr.get('bottom', 0) > 0 else -1

            line = f"{filename};"
            line += fmt(artery_tm) + ";"
            line += fmt(vein_tm) + ";"
            line += (f"{top_ratio:.4f}" if top_ratio > 0 else "-") + ";"
            line += fmt(artery_bm) + ";"
            line += fmt(vein_bm) + ";"
            line += (f"{btm_ratio:.4f}" if btm_ratio > 0 else "-") + ";\n"

            f.write(line)

    def _save_csv_rez2(self, csv_path: str):
        """
        Išsaugo _rez2.csv — C++ save_text2 formatas.
        Detalūs duomenys: skalė, OD, visi matavimai, klasifikacija.
        """
        with open(csv_path, 'w') as f:
            # Skalė ir OD
            f.write("Skale;OD x;OD y;OD r\n")
            f.write(f"{self.result.scale:.4f};{self.result.od_x};{self.result.od_y};{self.result.od_r}\n\n")

            # Visi matavimai
            classification = self.result.classification or {}
            classes = classification.get('classes', [])

            f.write("Numeris;Klase;Cx;Cy;Plotis;Kampas\n")
            for i, m in enumerate(self.result.measurements):
                cls = classes[i] if i < len(classes) else 0
                cls_name = {0: '-', 1: 'A_top', 2: 'V_top', 3: 'A_btm', 4: 'V_btm'}.get(cls, '-')
                f.write(f"{i};{cls_name};{m['x']:.1f};{m['y']:.1f};{m['width']:.4f};{m['angle']:.4f}\n")

            # Požymiai
            f.write("\nNumeris;Klase;Pozymiai\n")
            for i, m in enumerate(self.result.measurements):
                cls = classes[i] if i < len(classes) else 0
                feats = ";".join(f"{v:.4f}" for v in m.get('features', []))
                f.write(f"{i};{cls};{feats}\n")


def process_single_image(input_path: str, output_path: str,
                         progress_callback: Optional[Callable] = None) -> ProcessingResult:
    """
    Apdoroja vieną paveikslėlį.

    Args:
        input_path: Įvesties kelias
        output_path: Išvesties kelias
        progress_callback: Progreso callback

    Returns:
        ProcessingResult
    """
    processor = VesselProcessor()

    if progress_callback:
        processor.set_progress_callback(progress_callback)

    if not processor.load_image(input_path):
        return processor.result

    result = processor.run_full_processing()

    if result.success:
        # Išsaugoti rezultatus
        base_path = os.path.splitext(output_path)[0]
        processor.save_results(base_path)

    return result