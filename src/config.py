"""
config.py - Konfigūracija ir globalūs parametrai

Visi parametrai yra masteliuojami pagal paveiksliuko dydį.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =============================================================================
# KONSTANTOS
PI = np.pi
PI2 = 2.0 * PI
PIh = PI / 2.0


# =============================================================================
# GLOBALŪS PAVEIKSLĖLIAI
@dataclass
class ImageStore:
    """
    Saugo visus darbinius paveikslėlius.
    """
    # Originalus ir rezultatų paveikslėliai
    image: Optional[np.ndarray] = None           # uzkrautas paveiksliukas
    image_rez: Optional[np.ndarray] = None       # rezultatus atvaizduojantis paveiksliukas

    # Spalvų kanalai
    img_g: Optional[np.ndarray] = None           # zalio kanalo paveiksliukas
    img_g2: Optional[np.ndarray] = None
    img_r: Optional[np.ndarray] = None           # raudono kanalo paveiksliukas

    # Apdoroti paveikslėliai
    img_prep1: Optional[np.ndarray] = None       # pirminis apdorojimas kraujagysliu isgavimui
    img_prep2: Optional[np.ndarray] = None       # pirminis apdorojimas eksperto matavimams
    img_prep3: Optional[np.ndarray] = None
    img_prep4: Optional[np.ndarray] = None

    # Kraujagyslių paveikslėliai
    img_vsl: Optional[np.ndarray] = None         # isskirtas kraujaggysliu tinklas
    img_vsl2: Optional[np.ndarray] = None        # papildytas kraujagysliu tinklas
    img_vsl3: Optional[np.ndarray] = None        # matavimu taskai
    img_vsl4: Optional[np.ndarray] = None        # matavimu taskai
    img_vsl_thn: Optional[np.ndarray] = None     # suplonintas kraujaggysliu tinklas
    img_vsl_thn2: Optional[np.ndarray] = None    # suplonintas tinklas permatavimams

    # Kaukės
    img_mask: Optional[np.ndarray] = None        # kauke
    zero_mask: Optional[np.ndarray] = None
    odmask: Optional[np.ndarray] = None          # optinio disko kauke

    # Rezultatų atvaizdavimas
    img_rez_sc: Optional[np.ndarray] = None      # naudojamas rezultatams atvaizduoti

    def reset(self):
        """Išvalo visus paveikslėlius."""
        for attr in self.__dataclass_fields__:
            setattr(self, attr, None)


# =============================================================================
# OPTINIO DISKO PARAMETRAI
@dataclass
class OpticDiscParams:
    """Optinio disko koordinatės ir spindulys."""
    x: int = 0      # od_x
    y: int = 0      # od_y
    r: int = 0      # od_r


# =============================================================================
# SKALĖS PARAMETRAI
@dataclass
class ScaleParams:
    """
    Skalės parametrai.
    sc perskaičiuojamas setScale() funkcijoje.
    """
    sc: float = 1.0              # skalės parametras, perskaičiuojamas setScale() sigmoid formule
    pwd_sc: float = 12.0         # pusė profilio pločio prieš daugybą iš skalės
    pwd: float = 12.0            # pusė profilio pločio, perskaičiuojamas setScale()
    nstp: float = 100.0          # žingsnių skaičius profilyje
    stp: float = 0.10            # žingsnis kuriuo tiriamas profilis, perskaičiuojamas setScale()


# =============================================================================
# PROFILIO ANALIZĖS PARAMETRAI
@dataclass
class ProfileParams:
    """Profilio analizės parametrai."""
    navg: int = 1                # kiek kartų vidurkinamas profilis
    avg_step: int = 2            # kiek taškų į kiekvieną pusę imama vidurkinant
    fminpr: float = 0.1          # % į kiekvieną pusę nuo paspausto taško ieškant min
    thrp: float = 0.5            # paieškos slenkstis
    mmd: int = 15                # minimalus skirtumas tarp min ir max reikšmių


# =============================================================================
# KRAUJAGYSLĖS ANALIZĖS PARAMETRAI
@dataclass
class VesselParams:
    """Kraujagyslės analizės parametrai."""
    nvsl: int = 100              # kiek profilių imti vienam matavimui
    nvsl_min: int = 10           # kiek mažiausiai profilių turi būti išmatuota
    # Matavimai atliekami šiais atstumais nuo OD centro (kaip C++ odr_mult)
    # AVR zona vizualizacijai: [1.5×rOD; 3×rOD]
    odr_mult: List[float] = field(default_factory=lambda: [2.0, 2.5])
    pd1: float = 5.0             # atstumas tarp taškų ant apskritimo
    pd2: float = 0.5             # daugiklis


# =============================================================================
# MATAVIMŲ DUOMENYS
@dataclass
class MeasurementData:
    """
    Matavimų duomenų saugykla.
    """
    # Kiekvienas tipas: 1-viršutinė arterija, 2-viršutinė vena,
    #                   3-apatinė arterija, 4-apatinė vena
    v_rad: List[List[float]] = field(default_factory=lambda: [[], [], [], []])
    v_len: List[List[float]] = field(default_factory=lambda: [[], [], [], []])
    v_cx: List[List[float]] = field(default_factory=lambda: [[], [], [], []])
    v_cy: List[List[float]] = field(default_factory=lambda: [[], [], [], []])

    # Statistika
    mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    sd: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    var: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def clear(self, measure_type: int = None):
        """Išvalo matavimus."""
        if measure_type is None:
            for i in range(4):
                self.v_rad[i].clear()
                self.v_len[i].clear()
                self.v_cx[i].clear()
                self.v_cy[i].clear()
        else:
            idx = measure_type - 1
            if 0 <= idx < 4:
                self.v_rad[idx].clear()
                self.v_len[idx].clear()
                self.v_cx[idx].clear()
                self.v_cy[idx].clear()


# =============================================================================
# KRAUJAGYSLIŲ SAUGOJIMO PARAMETRAI (vslsave)
VSL_TYPE = 0      # kraujagyslės klasė
VSL_X = 1         # x koordinatė (centro)
VSL_Y = 2         # y koordinatė (centro)
VSL_R = 3         # kryptis
VSL_W = 4         # kraujagyslės plotis
VSL_M1 = 5        # a-a vidurkis
VSL_M2 = 6        # a-d vidurkis
VSL_M3 = 7        # d-c apatiniai vidurkis
VSL_M4 = 8        # c-d viršutiniai vidurkis
VSL_M5 = 9        # d-b vidurkis
VSL_M6 = 10       # vidurkis tarp d taškų
VSL_M7 = 11       # vidurkis tarp apatinių d taškų
VSL_MN1 = 12      # a-a minimumas
VSL_MX1 = 13      # a-a maksimumas
VSL_MX3 = 14      # d-c maksimumas
VSL_S1 = 15       # a-a vidurkis padalintas iš d-b vidurkio
# Raudonos spalvos parametrai (16-26)
VSL_M1R = 16
VSL_M2R = 17
VSL_M3R = 18
VSL_M4R = 19
VSL_M5R = 20
VSL_M6R = 21
VSL_M7R = 22
VSL_MN1R = 23
VSL_MX1R = 24
VSL_MX3R = 25
VSL_S1R = 26

PARS = 27  # parametrų skaičius
CLASS_SEL = [5, 6, 11, 16, 17, 22]  # klasifikacijai naudojami parametrai (žalias + raudonas kanalas)


# =============================================================================
# ATVAIZDAVIMO PARAMETRAI
@dataclass
class DisplayParams:
    """Atvaizdavimo nustatymai."""
    view_ratio: float = 1.0
    prepr: int = 1               # rodyti preprocess
    addvsl: int = 1              # rodyti kraujagysles
    addvslthn: int = 0           # rodyti suplonintas
    addod: int = 1               # rodyti optinį diską
    addrez: int = 1              # rodyti rezultatus


# =============================================================================
# EKSPERTO MATAVIMŲ PARAMETRAI
# NEPANAUDOTA
@dataclass
class ExpertMeasurements:
    """Eksperto matavimų duomenys."""
    from_x: int = 0
    to_x: int = 0
    from_y: int = 0
    to_y: int = 0

    exp_xf: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    exp_xt: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    exp_yf: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    exp_yt: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    exp_len: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


# =============================================================================
# PAGRINDINĖ KONFIGŪRACIJOS KLASĖ
class Config:
    """
    Pagrindinė konfigūracijos klasė.
    Apjungia visus parametrus ir paveikslėlius.
    """

    def __init__(self):
        self.images = ImageStore()
        self.optic_disc = OpticDiscParams()
        self.scale = ScaleParams()
        self.profile = ProfileParams()
        self.vessel = VesselParams()
        self.measurements = MeasurementData()
        self.display = DisplayParams()
        self.expert = ExpertMeasurements()

        # Kraujagyslių saugojimo masyvas
        self.vslsave: List[List[float]] = []

        # Tekstai
        self.save_text: str = ""
        self.save_text2: str = ""
        self.alert: str = ""
        self.form_out: str = ""

        # Failo informacija
        self.rawname: str = ""
        self.ext: str = ""

        # Matavimo tipas (1-4)
        self.measure_type: int = 0

        # Papildomi paveikslėliai
        self.mat_found: Optional[np.ndarray] = None
        self.mat_found2: Optional[np.ndarray] = None

        # Kraujagyslių saugojimas pagal tipą
        self.vsl_save_rad: List[List[float]] = [[], [], [], []]
        self.vsl_save_rad2: List[List[float]] = [[], [], [], []]
        self.vsl_save_len: List[List[float]] = [[], [], [], []]
        self.vsl_save_cx: List[List[float]] = [[], [], [], []]
        self.vsl_save_cy: List[List[float]] = [[], [], [], []]

    def setScale(self, img_mask: np.ndarray):
        """
        Nustato skalės parametrus pagal kaukės plotį.
            1. Skenuoja kaukę horizontaliai, randa min_m ir max_m
            2. wd = max_m - min_m (kaukės plotis pikseliais)
            3. sc = 4.0 / (1.0 + 20.0 * exp(-(wd / 500.0))) + wd / 3000.0
            4. pwd = pwd_sc * sc
            5. stp = 2 * pwd / nstp

        Args:
            img_mask: Kaukės paveikslėlis (CV_8UC1, 0 arba 255)
        """
        # Kaukės pločio apskaičiavimas (kaip C++)
        min_m = img_mask.shape[1]  # img_mask.cols
        max_m = 1

        for i in range(1, img_mask.shape[0]):
            lp = 0
            for j in range(1, img_mask.shape[1]):
                cp = int(img_mask[i, j])

                if lp == 0 and cp == 255 and min_m > j:
                    min_m = j
                elif lp == 255 and cp == 0 and max_m < j:
                    max_m = j

                lp = cp

        # Kaukės plotis
        wd = max_m - min_m

        # Sigmoid formulė
        self.scale.sc = 4.0 / (1.0 + 20.0 * np.exp(-(float(wd) / 500.0))) + float(wd) / 3000.0

        # Perskaičiuojame priklausomus parametrus
        self.scale.pwd = self.scale.pwd_sc * self.scale.sc   # pusė profilio pločio
        self.scale.stp = 2.0 * self.scale.pwd / self.scale.nstp  # profilio tyrimo žingsnis

    def reset(self):
        """Išvalo visą konfigūraciją."""
        self.images.reset()
        self.optic_disc = OpticDiscParams()
        self.measurements = MeasurementData()
        self.vslsave.clear()
        self.save_text = ""
        self.save_text2 = ""
        self.alert = ""
        self.measure_type = 0


# Globalus konfigūracijos objektas
config = Config()