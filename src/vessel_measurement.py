"""
vessel_measurement.py - Kraujagyslių matavimai ir klasifikacija
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

try:
    from config import config, PI, PI2, PIh, PARS, CLASS_SEL
except ImportError:
    from .config import config, PI, PI2, PIh, PARS, CLASS_SEL


def _lround(x: float) -> int:

    if x >= 0:
        return int(math.floor(x + 0.5))
    else:
        return int(math.ceil(x - 0.5))

# =============================================================================
# NUMBA OPTIMIZACIJOS (su fallback jei Numba nepasiekiama)

_NUMBA_AVAILABLE = False
_NUMBA_WARMED_UP = False

# Automatiškai išvalyti seną Numba cache (jei liko iš ankstesnių versijų)
import os as _os
import shutil as _shutil
_cache_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '__pycache__')
if _os.path.exists(_cache_dir):
    # Ištrinti tik Numba cache failus (*.nbi, *.nbc)
    for _f in _os.listdir(_cache_dir):
        if _f.endswith(('.nbi', '.nbc')):
            try:
                _os.remove(_os.path.join(_cache_dir, _f))
            except OSError:
                pass

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True

    @njit(cache=False)
    def _getProfilePointData_numba(img, point_x, point_y):
        """Numba optimizuota profilio taško reikšmė."""
        # C++ std::lround compatible: round half away from zero
        if point_x >= 0:
            pnt_x = int(math.floor(point_x + 0.5))
        else:
            pnt_x = int(math.ceil(point_x - 0.5))
        if point_y >= 0:
            pnt_y = int(math.floor(point_y + 0.5))
        else:
            pnt_y = int(math.ceil(point_y - 0.5))
        h, w = img.shape[0], img.shape[1]

        sumv = 0.0
        sumh = 0.0

        for i in range(-4, 5):
            for j in range(-4, 5):
                pnti_x = pnt_x + i
                pnti_y = pnt_y + j

                if 0 <= pnti_x < w and 0 <= pnti_y < h:
                    d = math.sqrt((point_x - pnti_x)**2 + (point_y - pnti_y)**2)
                    weight = 1.0 - (d / 3.0)
                    if weight < 0:
                        weight = 0.0

                    sumh += weight
                    val = float(img[pnti_y, pnti_x])
                    sumv += val * weight

        if sumh > 0:
            return sumv / sumh
        return 0.0

    @njit(cache=False)
    def _avgVector_numba(arr, navg, avg_step):
        """Numba optimizuotas vidurkinimas su IN-PLACE modifikacija."""
        n_len = len(arr)

        for _ in range(navg):
            for n in range(n_len):
                n1 = max(0, n - avg_step)
                n2 = min(n_len - 1, n + avg_step)

                sumn = 0.0
                for k in range(n1, n2 + 1):
                    sumn += arr[k]

                arr[n] = sumn / (n2 - n1 + 1)

        return arr

    @njit(parallel=True, cache=False)
    def _getProfileData_numba(img, points):
        """Numba PARALELIZUOTAS profilio duomenų gavimas."""
        n_points = points.shape[0]
        result = np.zeros(n_points, dtype=np.float64)

        for i in prange(n_points):
            result[i] = _getProfilePointData_numba(img, points[i, 0], points[i, 1])

        return result

    @njit(cache=False)
    def _getProfilePoints_numba(x, y, rd, pwd, stp):
        """Numba optimizuotas profilio taškų generavimas."""
        n_points = int(2 * pwd / stp) + 1
        points = np.zeros((n_points, 2), dtype=np.float64)

        cos_rd = math.cos(rd)
        sin_rd = math.sin(rd)

        r = -pwd
        idx = 0
        while r <= pwd and idx < n_points:
            points[idx, 0] = x + (r * cos_rd)
            points[idx, 1] = y + (r * sin_rd)
            r += stp
            idx += 1

        return points[:idx]

    def _warmup_numba():
        """Sukompiliuoja Numba funkcijas."""
        global _NUMBA_WARMED_UP
        if _NUMBA_WARMED_UP:
            return

        test_img = np.zeros((50, 50), dtype=np.uint8)
        _ = _getProfilePointData_numba(test_img, 25.0, 25.0)
        points = _getProfilePoints_numba(25.0, 25.0, 0.0, 5.0, 0.5)
        _ = _getProfileData_numba(test_img, points)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        _ = _avgVector_numba(vec, 1, 1)
        _NUMBA_WARMED_UP = True

except ImportError:
    _NUMBA_AVAILABLE = False


# =============================================================================
# OPTIMIZUOTOS PAGALBINĖS FUNKCIJOS
# =============================================================================

def avgVector(vec: List[float], navg: int, avg_step: int) -> List[float]:
    """
    Suvidurija vektorių naudojant slenkantį langą.

    Args:
        vec: Pradinis vektorius
        navg: Kiek kartų vidurkiname
        avg_step: Kiek taškų į kiekvieną pusę imame

    Returns:
        Suvidurkintas vektorius
    """
    arr = np.array(vec, dtype=np.float64)

    if _NUMBA_AVAILABLE:
        _warmup_numba()
        return _avgVector_numba(arr, navg, avg_step).tolist()

    # Fallback
    n_len = len(arr)
    for _ in range(navg):
        for n in range(n_len):
            n1 = max(0, n - avg_step)
            n2 = min(n_len - 1, n + avg_step)
            sumn = np.sum(arr[n1:n2 + 1])
            arr[n] = sumn / (n2 - n1 + 1)

    return arr.tolist()


def getDistPoints(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Atstumas tarp dviejų taškų."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def distMat(crdx: List[float], crdy: List[float]) -> np.ndarray:
    """
    Atstumų matrica -  su numpy broadcasting.
    """
    x = np.array(crdx, dtype=np.float32)
    y = np.array(crdy, dtype=np.float32)

    # Broadcasting: (n,1) - (1,n) = (n,n)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]

    return np.sqrt(dx**2 + dy**2).astype(np.float32)


def getNN(distm: np.ndarray, idx: int, nn: int, sz: int) -> List[int]:
    """
    Artimiausi kaimynai su numpy
    """
    distances = distm[idx, :]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:nn].tolist()


def getProfilePointData(img: np.ndarray, point: Tuple[float, float]) -> float:
    """
    Profilio taško reikšmė - OPTIMIZUOTA su numpy.
    """
    pnt_x = _lround(point[0])
    pnt_y = _lround(point[1])
    h, w = img.shape[:2]

    # Sukurti koordinačių tinklelį
    i_range = np.arange(-4, 5)
    j_range = np.arange(-4, 5)
    ii, jj = np.meshgrid(i_range, j_range)

    pnti_x = pnt_x + ii
    pnti_y = pnt_y + jj

    # Ribų kaukė
    valid = (pnti_x >= 0) & (pnti_x < w) & (pnti_y >= 0) & (pnti_y < h)

    if not np.any(valid):
        return 0.0

    # Atstumai ir svoriai
    d = np.sqrt((point[0] - pnti_x)**2 + (point[1] - pnti_y)**2)
    weights = np.maximum(0, 1 - d / 3) * valid

    # Gauti reikšmes su clipping
    pnti_x_safe = np.clip(pnti_x, 0, w - 1)
    pnti_y_safe = np.clip(pnti_y, 0, h - 1)
    vals = img[pnti_y_safe, pnti_x_safe].astype(np.float64)

    sumh = np.sum(weights)
    if sumh > 0:
        return np.sum(vals * weights) / sumh
    return 0.0


def getProfilePoints(x: float, y: float, rd: float,
                     pwd: float, stp: float) -> List[Tuple[float, float]]:
    """
    Surenka profilio taškų koordinates.

    Args:
        x, y: Centro koordinatės
        rd: Kryptis (radianais)
        pwd: Pusė profilio pločio
        stp: Žingsnis

    Returns:
        Taškų koordinačių sąrašas
    """
    points = []

    r = -pwd
    while r <= pwd:
        point_x = x + (r * math.cos(rd))
        point_y = y + (r * math.sin(rd))
        points.append((point_x, point_y))
        r += stp

    return points


def getProfileData(img: np.ndarray,
                   points: List[Tuple[float, float]]) -> List[float]:
    """
    Surenka profilio duomenis pagal koordinates.

    Args:
        img: Pilkas paveikslėlis
        points: Taškų koordinatės

    Returns:
        Profilio reikšmių sąrašas
    """
    if _NUMBA_AVAILABLE and len(points) > 10:
        _warmup_numba()
        # Konvertuoti į numpy array
        points_arr = np.array(points, dtype=np.float64)
        return _getProfileData_numba(img, points_arr).tolist()

    # Fallback
    profile = []
    for point in points:
        profile.append(getProfilePointData(img, point))
    return profile


def getProfileAnalysisPoints(valv: List[float], mmd: float,
                             thrp: float, fminpr: float = 0.1) -> List[int]:
    """
    Analizuoja profilį ir randa svarbius taškus.

    Grąžina taškus:
    [nla, nra, nldb, nrdb, nle, nre, nldt, nrdt, nlb, nrb]

    Args:
        valv: Profilio reikšmės
        mmd: Minimalus skirtumas tarp min ir max
        thrp: Slenksčio koeficientas
        fminpr: Minimumo paieškos intervalas nuo centro

    Returns:
        Analizės taškų indeksai arba tuščias sąrašas jei nerasta
    """
    valsz = len(valv)
    if valsz < 10:
        return []

    valszd = float(valsz)

    # Inicializuoti kintamuosius
    min_val = 255.0
    max_val = 0.0
    nmin = valsz // 2

    n1 = _lround(valszd * 0.5 - valszd * fminpr)
    n2 = _lround(valszd * 0.5 + valszd * fminpr)

    for n in range(max(0, n1), min(valsz, n2 + 1)):
        if valv[n] <= min_val:
            min_val = valv[n]
            nmin = n

    for n in range(valsz):
        if valv[n] >= max_val:
            max_val = valv[n]

    if max_val - min_val <= mmd:
        return []

    thr = min_val + (max_val - min_val) * thrp

    if nmin < valsz // 2:
        nl = nmin
        nr = valsz // 2
    else:
        nl = valsz // 2
        nr = nmin

    nla = nl
    nra = nr
    nl3 = nl
    nr3 = nr

    # ==========================================================================
    # KAIRĖ PUSĖ
    n = nl
    for n in range(nl, 0, -1):
        if valv[n] >= valv[n - 1] and valv[n] > thr:
            break
    nlb = n
    nl = n
    nl2 = nl

    # Inicializuoti reikšmes
    nldb = nl
    nldt = nl
    nle = nl
    nlc = nl

    if valv[nl] <= thr:
        nl = -1
        nle = -1
    else:
        # ieškoti didžiausio nusileidimo
        if nl3 > nl2:
            dif_mean = 0.0
            dif_max = valv[nl3] - valv[nl3 - 1] if nl3 > 0 else 0
            dif_max_n = nl3

            for n in range(nl3, nl2, -1):
                if n > 0:
                    dif = valv[n - 1] - valv[n]
                    dif_mean += dif
                    if dif_max <= dif:
                        dif_max = dif
                        dif_max_n = n

            nlc = dif_max_n

            count = nl3 - nl2 - 1
            if count > 0:
                dif_mean = dif_mean / count

                dif_sd = 0.0
                for n in range(nl3, nl2, -1):
                    if n > 0:
                        dif = valv[n - 1] - valv[n]
                        dif_sd += (dif - dif_mean) ** 2

                dif_sd = math.sqrt(dif_sd / count) if count > 0 else 0

                # Rasti nldb
                for n in range(nl3, nl2, -1):
                    if n > 0:
                        if (valv[n - 1] - valv[n]) >= (dif_max - dif_sd):
                            dif_max_n = n
                            break

                nldb = n
                nl = dif_max_n

                # Rasti nldt
                for n in range(nl2 - 1, nl3):
                    if n > 0 and n < valsz:
                        if (valv[n - 1] - valv[n]) >= (dif_max - dif_sd):
                            dif_max_n = n
                            break

                nldt = n
                if nl != dif_max_n:
                    nl = int((dif_max_n + nl) * 0.5)
                nle = nl

    # ==========================================================================
    # DEŠINĖ PUSĖ
    #  ieškoti dešinio krašto
    for n in range(nr, valsz - 1):
        if valv[n] >= valv[n + 1] and valv[n] > thr:
            break
    nrb = n
    nr = n
    nr2 = nr

    # Inicializuoti reikšmes
    nrdb = nr
    nrdt = nr
    nre = nr
    nrc = nr

    if valv[nr] <= thr:
        nr = -1
        nre = -1
    else:
        #  ieškoti didžiausio nusileidimo
        if nr2 > nr3:
            dif_mean = 0.0
            dif_max = valv[nr3] - valv[nr3 + 1] if nr3 < valsz - 1 else 0
            dif_max_n = nr3

            for n in range(nr3, nr2):
                if n < valsz - 1:
                    dif = valv[n + 1] - valv[n]
                    dif_mean += dif
                    if dif_max <= dif:
                        dif_max = dif
                        dif_max_n = n

            nrc = dif_max_n

            count = nr2 - nr3 - 1
            if count > 0:
                dif_mean = dif_mean / count

                dif_sd = 0.0
                for n in range(nr3, nr2):
                    if n < valsz - 1:
                        dif = valv[n + 1] - valv[n]
                        dif_sd += (dif - dif_mean) ** 2

                dif_sd = math.sqrt(dif_sd / count) if count > 0 else 0

                # Rasti nrdb
                for n in range(nr3, nr2):
                    if n < valsz - 1:
                        if (valv[n + 1] - valv[n]) >= (dif_max - dif_sd):
                            dif_max_n = n
                            break

                nrdb = dif_max_n
                nr = dif_max_n

                # Rasti nrdt
                for n in range(nr2 - 1, nr3, -1):
                    if n < valsz - 1:
                        if (valv[n + 1] - valv[n]) >= (dif_max - dif_sd):
                            dif_max_n = n
                            break

                nrdt = dif_max_n
                if nr != dif_max_n:
                    nr = int((dif_max_n + nr) * 0.5)
                nre = nr

    nle = nl if nl != -1 else nla
    nre = nr if nr != -1 else nra

    #  grąžinti taškus
    points = [nla, nra, nldb, nrdb, nle, nre, nldt, nrdt, nlb, nrb]
    return points


def getProfileFeatures(valv: List[float], points: List[int]) -> List[float]:
    """
    Apskaičiuoja profilio požymius klasifikavimui.

    Args:
        valv: Profilio reikšmės
        points: Analizės taškai [al, ar, d1l, d1r, cl, cr, d2l, d2r, bl, br]

    Returns:
        Požymių sąrašas [m1, m2, m3, m4, m5, m6, m7, mn1, mx1, mx3, s1]
    """
    if len(points) < 10 or len(valv) == 0:
        return [0.0] * 11

    features = []

    # m1: vidurkis al-ar

    sm, nm = 0.0, 0
    smn = None
    smx = None

    for n in range(points[0], points[1] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1

            if smn is None:
                smn = valv[n]
                smx = valv[n]
            else:

                if smn > valv[n]:
                    smn = valv[n]

                if smx < valv[n]:
                    smx = valv[n]

    m1 = sm / nm if nm > 0 else 0
    mn1 = smn if smn is not None else 0.0  # minimumas al-ar
    mx1 = smx if smx is not None else 0.0  # maksimumas al-ar

    features.append(m1)  # 0: m1

    # m2: vidurkis a-d (kraštai)
    sm, nm = 0.0, 0
    for n in range(points[2], points[0] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1
    for n in range(points[1], points[3] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1

    m2 = sm / nm if nm > 0 else 0
    features.append(m2)  # 1: m2

    # m3: vidurkis d1-c (apatiniai)
    sm, nm = 0.0, 0
    mx3 = None
    for n in range(points[4], points[2] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1
            if mx3 is None:
                mx3 = valv[n]
            elif mx3 < valv[n]:
                mx3 = valv[n]
    for n in range(points[3], points[5] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1
            if mx3 is None:
                mx3 = valv[n]
            elif mx3 < valv[n]:
                mx3 = valv[n]

    m3 = sm / nm if nm > 0 else 0
    mx3 = mx3 if mx3 is not None else 0.0
    features.append(m3)  # 2: m3

    # m4: vidurkis c-d (viršutiniai)
    sm, nm = 0.0, 0
    for n in range(points[6], points[4] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1
    for n in range(points[5], points[7] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1

    m4 = sm / nm if nm > 0 else 0
    features.append(m4)  # 3: m4

    # m5: vidurkis d-b
    sm, nm = 0.0, 0
    for n in range(points[8], points[6] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1
    for n in range(points[7], points[9] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1

    m5 = sm / nm if nm > 0 else 0
    features.append(m5)  # 4: m5

    # m6: vidurkis d1-d2
    # points[6] -> points[2] ir points[3] -> points[7]
    sm, nm = 0.0, 0
    for n in range(points[6], points[2] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1
    for n in range(points[3], points[7] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1

    m6 = sm / nm if nm > 0 else 0
    features.append(m6)  # 5: m6

    # m7: vidurkis d1l-d1r
    # points[2] -> points[3]
    sm, nm = 0.0, 0
    for n in range(points[2], points[3] + 1):
        if 0 <= n < len(valv):
            sm += valv[n]
            nm += 1

    m7 = sm / nm if nm > 0 else 0
    features.append(m7)  # 6: m7

    # Pridėti min, max, s1
    features.append(mn1)  # 7: mn1
    features.append(mx1)  # 8: mx1
    features.append(mx3)  # 9: mx3

    # s1: m5 / m1 santykis  features[4] / features[0])
    s1 = m5 / m1 if m1 > 0 else 0
    features.append(s1)  # 10: s1

    return features


# =============================================================================
# MATAVIMO FUNKCIJOS

def isMeasurePointValid(x: float, y: float, img_mask: np.ndarray,
                        img_vsl_thn: np.ndarray,
                        check_skeleton: bool = True,
                        mat_found: np.ndarray = None) -> bool:
    """
    Patikrina ar matavimo taškas yra validus.

    tikrina: ribas (2*pwd) + mat_found.
    Skeleto tikrinimas — tik pradiniams taškams (measure_od_points).
    mat_found tikrinimas — walk žingsniuose

    Args:
        x, y: Taško koordinatės
        img_mask: Akies dugno kaukė
        img_vsl_thn: Suplonintas kraujagyslių paveikslėlis
        check_skeleton: Ar tikrinti skeletą (True tik pradiniams taškams)
        mat_found: Jau išmatuotų sričių kaukė

    Returns:
        True jei taškas validus
    """
    h, w = img_mask.shape[:2]
    ix, iy = int(x), int(y)

    # Patikrinti ribas
    if ix < 5 or ix >= w - 5 or iy < 5 or iy >= h - 5:
        return False

    # Walk sustoja ties zonos ribomis ir jau išmatuotais kontūrais
    if mat_found is not None:
        if mat_found[iy, ix] > 0:
            return False

    # Patikrinti ar taškas yra kaukės viduje
    if img_mask[iy, ix] == 0:
        return False

    # Skeleto tikrinimas — tik pradiniams taškams
    if check_skeleton:
        region = img_vsl_thn[iy-2:iy+3, ix-2:ix+3]
        if cv2.countNonZero(region) == 0:
            return False

    return True


# =============================================================================
# SKELETO SEKIMO FUNKCIJOS

def getPointVal(x: int, y: int, dir: int, thn: np.ndarray) -> bool:
    """
    Patikrina ar kaimynas nurodytoje kryptyje yra baltas.

    Krypčių schema:
    1=viršus, 2=dešinė, 3=apačia, 4=kairė
    5=viršus-dešinė, 6=apačia-dešinė, 7=apačia-kairė, 8=viršus-kairė
    """
    h, w = thn.shape[:2]

    if dir == 1:
        ny, nx = y - 1, x
    elif dir == 2:
        ny, nx = y, x + 1
    elif dir == 3:
        ny, nx = y + 1, x
    elif dir == 4:
        ny, nx = y, x - 1
    elif dir == 5:
        ny, nx = y - 1, x + 1
    elif dir == 6:
        ny, nx = y + 1, x + 1
    elif dir == 7:
        ny, nx = y + 1, x - 1
    elif dir == 8:
        ny, nx = y - 1, x - 1
    else:
        return False

    if 0 <= nx < w and 0 <= ny < h:
        return thn[ny, nx] > 0
    return False


def getNextDirX(x: int, dir: int) -> int:

    if dir == 1: return x
    elif dir == 2: return x + 1
    elif dir == 3: return x
    elif dir == 4: return x - 1
    elif dir == 5: return x + 1
    elif dir == 6: return x + 1
    elif dir == 7: return x - 1
    elif dir == 8: return x - 1
    else: return x


def getNextDirY(y: int, dir: int) -> int:

    if dir == 1: return y - 1
    elif dir == 2: return y
    elif dir == 3: return y + 1
    elif dir == 4: return y
    elif dir == 5: return y - 1
    elif dir == 6: return y + 1
    elif dir == 7: return y + 1
    elif dir == 8: return y - 1
    else: return y


def getNextPointDir(x: int, y: int, dir: int, thn: np.ndarray) -> Tuple[int, int, int]:
    """
    Randa kitą tašką skelete pagal esamą kryptį.

    Returns:
        (xn, yn, dirn) - naujas taškas ir nauja kryptis, arba dirn=0 jei nerasta
    """
    # Surinkti kaimynų informaciją
    d = [0] * 9
    for i in range(1, 9):
        d[i] = 1 if getPointVal(x, y, i, thn) else 0

    dirn = 0

    # Tikrinti pagal esamą kryptį
    if dir == 1:
        if d[1] + d[2] + d[4] > 1 or d[5] + d[8] > 1 or d[1] + d[2] + d[4] + d[5] + d[8] == 0:
            dirn = 0
        else:
            if d[1] == 1: dirn = 1
            elif d[2] == 1: dirn = 2
            elif d[4] == 1: dirn = 4
            elif d[5] == 1: dirn = 5
            elif d[8] == 1: dirn = 8
    elif dir == 2:
        if d[2] + d[3] + d[4] > 1 or d[5] + d[6] > 1 or d[2] + d[3] + d[1] + d[6] + d[5] == 0:
            dirn = 0
        else:
            if d[2] == 1: dirn = 2
            elif d[3] == 1: dirn = 3
            elif d[1] == 1: dirn = 1
            elif d[6] == 1: dirn = 6
            elif d[5] == 1: dirn = 5
    elif dir == 3:
        if d[3] + d[4] + d[2] > 1 or d[6] + d[7] > 1 or d[3] + d[4] + d[2] + d[6] + d[7] == 0:
            dirn = 0
        else:
            if d[3] == 1: dirn = 3
            elif d[4] == 1: dirn = 4
            elif d[2] == 1: dirn = 2
            elif d[7] == 1: dirn = 7
            elif d[6] == 1: dirn = 6
    elif dir == 4:
        if d[4] + d[3] + d[1] > 1 or d[7] + d[8] > 1 or d[4] + d[3] + d[1] + d[7] + d[8] == 0:
            dirn = 0
        else:
            if d[4] == 1: dirn = 4
            elif d[3] == 1: dirn = 3
            elif d[1] == 1: dirn = 1
            elif d[7] == 1: dirn = 7
            elif d[8] == 1: dirn = 8
    elif dir == 5:
        if d[1] + d[2] > 1 or d[5] + d[6] + d[8] > 1 or d[1] + d[2] + d[5] + d[6] + d[8] == 0:
            dirn = 0
        else:
            if d[5] == 1: dirn = 5
            elif d[1] == 1: dirn = 1
            elif d[2] == 1: dirn = 2
            elif d[8] == 1: dirn = 8
            elif d[6] == 1: dirn = 6
    elif dir == 6:
        if d[2] + d[3] > 1 or d[5] + d[6] + d[7] > 1 or d[2] + d[3] + d[5] + d[6] + d[7] == 0:
            dirn = 0
        else:
            if d[6] == 1: dirn = 6
            elif d[2] == 1: dirn = 2
            elif d[3] == 1: dirn = 3
            elif d[5] == 1: dirn = 5
            elif d[7] == 1: dirn = 7
    elif dir == 7:
        if d[3] + d[4] > 1 or d[6] + d[7] + d[8] > 1 or d[3] + d[4] + d[6] + d[7] + d[8] == 0:
            dirn = 0
        else:
            if d[7] == 1: dirn = 7
            elif d[3] == 1: dirn = 3
            elif d[4] == 1: dirn = 4
            elif d[6] == 1: dirn = 6
            elif d[8] == 1: dirn = 8
    elif dir == 8:
        if d[4] + d[1] > 1 or d[7] + d[8] + d[5] > 1 or d[4] + d[1] + d[7] + d[8] + d[5] == 0:
            dirn = 0
        else:
            if d[8] == 1: dirn = 8
            elif d[4] == 1: dirn = 4
            elif d[1] == 1: dirn = 1
            elif d[7] == 1: dirn = 7
            elif d[5] == 1: dirn = 5

    xn = getNextDirX(x, dirn)
    yn = getNextDirY(y, dirn)

    return xn, yn, dirn


def findTreePoints(x: int, y: int, stps: int, szs: int,
                   thn: np.ndarray) -> float:
    """
    Seka skeletą topologiškai (pikselis po pikselio) ir grąžina kryptį.

    Args:
        x, y: Pradinio taško koordinatės
        stps: Kiek žingsnių sekti kiekvien pusę
        szs: Paieškos spindulis artimiausiam taškui
        thn: Suplonintas kraujagyslių paveikslėlis

    Returns:
        Kryptis radianais arba 9999 jei nerasta
    """
    h, w = thn.shape[:2]

    # Patikrinti ribas
    if x - szs - 1 < 0 or y - szs - 1 < 0:
        return 9999.0
    if x + szs + 1 >= w or y + szs + 1 >= h:
        return 9999.0

    # Rasti artimiausią skeleto tašką (su svoriu pagal atstumą)
    find = thn[y - szs:y + szs + 1, x - szs:x + szs + 1].astype(np.float64)

    for i in range(-szs, szs + 1):
        for j in range(-szs, szs + 1):
            dist = math.sqrt(i*i + j*j)
            if dist > 0:
                find[i + szs, j + szs] = find[i + szs, j + szs] / (1 + dist)

    # Rasti maksimumą
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(find)

    if max_val <= 0:
        return 9999.0

    # Perskaičiuoti koordinates
    x = x + max_loc[0] - szs
    y = y + max_loc[1] - szs

    # Patikrinti ribas
    if x <= 0 or y <= 0 or x >= w - 1 or y >= h - 1:
        return 9999.0

    #  Rasti kryptis (iki 3)
    dir1, dir2, dir3 = 0, 0, 0

    for dir in range(1, 9):
        if getPointVal(x, y, dir, thn):
            if dir1 == 0:
                dir1 = dir
            elif dir2 == 0:
                dir2 = dir
            elif dir3 == 0:
                dir3 = dir
            else:
                # Per daug krypčių - susikirtimas
                dir1, dir2, dir3 = 0, 0, 0
                break

    x1, y1 = x, y
    x2, y2 = x, y

    if dir1 > 0:
        # Sekti pirmąja kryptimi
        x1 = getNextDirX(x, dir1)
        y1 = getNextDirY(y, dir1)

        xn, yn = x1, y1
        dir_from = dir1

        for i in range(1, stps // 2 + 1):
            xn, yn, dir_to = getNextPointDir(x1, y1, dir_from, thn)

            if dir_to > 0:
                x1 = xn
                y1 = yn
                dir_from = dir_to
            else:
                break

        #  Atmesti priešingą kryptį
        if dir2 != 0:
            # Krypčių konflikto tikrinimas
            opposite_dirs = {
                1: [5, 8], 2: [5, 6], 3: [6, 7], 4: [7, 8],
                5: [1, 2], 6: [2, 3], 7: [3, 4], 8: [1, 4]
            }

            if dir1 in opposite_dirs:
                if dir2 in opposite_dirs[dir1]:
                    dir2 = dir3
                if dir2 in opposite_dirs.get(dir1, []):
                    dir2 = 0

        #  Sekti antrąja kryptimi
        if dir2 != 0:
            x2 = getNextDirX(x, dir2)
            y2 = getNextDirY(y, dir2)

            xn, yn = x2, y2
            dir_from = dir2

            for i in range(1, stps // 2 + 1):
                xn, yn, dir_to = getNextPointDir(x2, y2, dir_from, thn)

                if dir_to > 0:
                    x2 = xn
                    y2 = yn
                    dir_from = dir_to
                else:
                    break

    #  Grąžinti kryptį
    if x1 == x2 and y1 == y2:
        return 9999.0
    else:
        return math.atan2(y1 - y2, x1 - x2)


def measureVessel7_iteration(x: float, y: float, rad_from: float, rad_to: float,
                              img_g: np.ndarray, img_g2: np.ndarray, img_r: np.ndarray,
                              img_vsl_thn: np.ndarray, img_mask: np.ndarray,
                              sc: float, pwd: float, stp: float,
                              mmd: float, thrp: float,
                              mat_found: np.ndarray = None) -> Tuple[float, float, float, float, List[float]]:
    """
    Atlieka vieną matavimą su kampo paieška nurodytu diapazonu.

    Returns:
        (rx, ry, rrad, rlen, mmi) - centro koordinatės, kryptis, plotis, požymiai
    """
    rlen = 0.0
    rx, ry, rrad = 0.0, 0.0, 0.0
    mmi = []

    if not isMeasurePointValid(x, y, img_mask, img_vsl_thn, check_skeleton=False, mat_found=mat_found):
        return rx, ry, rrad, rlen, mmi

    rby = PI / 180.0  # kas 1 laipsni
    rdto = int(abs(rad_to - rad_from) / rby)

    v_rad = []
    v_r_len = []
    v_r_pf = []
    v_r_pt = []
    points_all = []
    v_vals = []

    # Eiti per visus kampus
    for rdi in range(rdto + 1):
        if rdi % 2 == 1:
            rd = (rad_from + rad_to) / 2.0 + math.ceil(rdi / 2.0) * rby
        else:
            rd = (rad_from + rad_to) / 2.0 - math.ceil(rdi / 2.0) * rby

        v_rad.append(rd)

        # Gauti profilio taškus
        v_points = getProfilePoints(x, y, rd, pwd, stp)

        # Profilio duomenys iš img_g2 (su centrinio reflekso panaikinimu)
        v_val = getProfileData(img_g2, v_points)
        v_val = avgVector(v_val, 1, 2)
        v_vals.append(v_val)

        # Profilio analizės taškai
        points = getProfileAnalysisPoints(v_val, mmd, thrp)
        points_all.append(points)

        if points:
            nle = points[4]
            nre = points[5]

            if nle >= 0 and nre >= 0 and nle < len(v_points) and nre < len(v_points):
                cr_len = getDistPoints(v_points[nle], v_points[nre])
                if cr_len > 0 and cr_len < pwd:
                    v_r_len.append(cr_len)
                    v_r_pf.append(v_points[nle])
                    v_r_pt.append(v_points[nre])
                else:
                    v_r_len.append(-1)
                    v_r_pf.append((-1, -1))
                    v_r_pt.append((-1, -1))
            else:
                v_r_len.append(-1)
                v_r_pf.append((-1, -1))
                v_r_pt.append((-1, -1))
        else:
            v_r_len.append(-1)
            v_r_pf.append((-1, -1))
            v_r_pt.append((-1, -1))

    # Rasti minimalų plotį
    min_len = -1
    minn = 0

    for n in range(len(v_r_len)):
        if v_r_len[n] > sc * 0.1:
            if min_len == -1 or min_len > v_r_len[n]:
                minn = n
                min_len = v_r_len[n]

    if min_len > -1:
        # Rasti visus kampus su mažiausiu pločiu
        minns = []
        for n in range(len(v_r_len)):
            if v_r_len[n] > sc * 0.1:
                if v_r_len[n] <= min_len * 1.0:
                    minns.append(n)

        n = minns[len(minns) // 2]

        pf = v_r_pf[n]
        pt = v_r_pt[n]

        rx = (pf[0] + pt[0]) / 2.0
        ry = (pf[1] + pt[1]) / 2.0
        rrad = v_rad[n]
        rlen = v_r_len[n]

        # Gauti požymius
        v_points = getProfilePoints(x, y, rrad, pwd, stp)

        # Žalias kanalas (be centrinio reflekso panaikinimo)
        v_val_green = getProfileData(img_g, v_points)
        v_val_green = avgVector(v_val_green, 1, 2)
        ft_green = getProfileFeatures(v_val_green, points_all[n])

        # Raudonas kanalas
        v_val_red = getProfileData(img_r, v_points)
        v_val_red = avgVector(v_val_red, 1, 2)
        ft_red = getProfileFeatures(v_val_red, points_all[n])

        mmi = ft_green + ft_red

    return rx, ry, rrad, rlen, mmi


def measureVesselPoint(x: float, y: float, img_g: np.ndarray, img_g2: np.ndarray,
                       img_r: np.ndarray, img_mask: np.ndarray, img_vsl_thn: np.ndarray,
                       sc: float, pwd: float, stp: float,
                       mmd: float, thrp: float,
                       mat_found: np.ndarray = None) -> Tuple[List[float], float, float]:
    """
    Matuoja kraujagyslę viename taške.

    Algoritmas:
    1. findTreePoints → pradinė kryptis
    2. 3× measureVessel7_iteration (pradinis + 2 patikslinimai)
    3. Ėjimas abiem kryptimis išilgai kraujagyslės (nvsl/2 žingsnių)
    4. Kampų normalizavimas (v_rad2: 0-PI sritis)
    5. Outlier aptikimas (>1 SD nuo vidurkio)
    6. Outlier pakeitimas avgVector versijomis + permatavimas
    7. Galinis avgVector v_len, v_cx, v_cy
    8. Kokybės patikra (sd, rsd, count, mean ribos)

    Returns:
        tuple: (požymiai, plotis, kryptis) arba ([], 0, 0) jei nepavyko
    """
    if not isMeasurePointValid(x, y, img_mask, img_vsl_thn, mat_found=mat_found):
        return [], 0.0, 0.0, None


    nvsl = 100 #30 !!! galimos probleminės vietos
    nvsl_min = 10 #5 !!!
    mstp = 2.0  # žingsnis išilgai kraujagyslės

    PI2 = 2.0 * PI

    # =====================================================================
    # 1. Rasti pradinę kryptį per findTreePoints
    tree_rad = findTreePoints(int(x), int(y), int(sc * 2), int(sc * 4), img_vsl_thn)

    rd = 0.0
    rdpm = PI / 18.0  #  numatytasis +-10 laipsniu

    if tree_rad != 9999:
        #rd = tree_rad - PIh; rdpm = PI / 30.0;
        rd = tree_rad - PIh
        rdpm = PI / 30.0  # +-6 laipsniai

    # =====================================================================
    # 2. Pradinis matavimas
    if tree_rad != 9999:
        rx1, ry1, rrad1, rlen1, mmi = measureVessel7_iteration(
            x, y, rd - rdpm, rd + rdpm,
            img_g, img_g2, img_r, img_vsl_thn, img_mask,
            sc, pwd, stp, mmd, thrp, mat_found=mat_found
        )
    else:
        rx1, ry1, rrad1, rlen1, mmi = measureVessel7_iteration(
            x, y, 0, PI,
            img_g, img_g2, img_r, img_vsl_thn, img_mask,
            sc, pwd, stp, mmd, thrp, mat_found=mat_found
        )
        # rdpm lieka PI/18 — naudojamas patikslinimui ir ėjimui

    if rlen1 <= 0:
        return [], 0.0, 0.0, None

    # =====================================================================
    # 3. Dvigubas patikslinimas
    # Naudoja tą patį rdpm (PI/30 jei medis, PI/18 jei ne)
    # =====================================================================
    rx1, ry1, rrad1, rlen1, mmi = measureVessel7_iteration(
        rx1, ry1, rrad1 - rdpm, rrad1 + rdpm,
        img_g, img_g2, img_r, img_vsl_thn, img_mask,
        sc, pwd, stp, mmd, thrp, mat_found=mat_found
    )

    if rlen1 <= 0:
        return [], 0.0, 0.0, None

    #antrasis patikslinimas
    rx1, ry1, rrad1, rlen1, mmi = measureVessel7_iteration(
        rx1, ry1, rrad1 - rdpm, rrad1 + rdpm,
        img_g, img_g2, img_r, img_vsl_thn, img_mask,
        sc, pwd, stp, mmd, thrp, mat_found=mat_found
    )

    if rlen1 <= 0:
        return [], 0.0, 0.0, None

    # =====================================================================
    # 4. Surinkti matavimus einant abiem kryptimis
    v_rad = [rrad1]
    v_rad2 = [rrad1]
    v_len = [rlen1]
    v_cx = [rx1]
    v_cy = [ry1]
    mm = [mmi]

    rrad = rrad1
    rx = rx1
    ry = ry1

    # Eiti į vieną pusę
    for i in range(1, nvsl // 2 + 1):
        nx = rx + (sc * mstp * math.cos(rrad + PIh))
        ny = ry + (sc * mstp * math.sin(rrad + PIh))

        rx_new, ry_new, rrad_new, rlen_new, mmi_new = measureVessel7_iteration(
            nx, ny, rrad - rdpm, rrad + rdpm,
            img_g, img_g2, img_r, img_vsl_thn, img_mask,
            sc, pwd, stp, mmd, thrp, mat_found=mat_found
        )

        if rlen_new <= 0:
            break

        v_rad.append(rrad_new)
        v_rad2.append(rrad_new)
        v_len.append(rlen_new)
        v_cx.append(rx_new)
        v_cy.append(ry_new)
        mm.append(mmi_new)

        rx = rx_new
        ry = ry_new
        rrad = rrad_new

    # Grįžti ir eiti į kitą pusę
    rrad = rrad1
    rx = rx1
    ry = ry1

    for i in range(1, nvsl // 2):
        nx = rx + (sc * mstp * math.cos(rrad - PIh))
        ny = ry + (sc * mstp * math.sin(rrad - PIh))

        rx_new, ry_new, rrad_new, rlen_new, mmi_new = measureVessel7_iteration(
            nx, ny, rrad - rdpm, rrad + rdpm,
            img_g, img_g2, img_r, img_vsl_thn, img_mask,
            sc, pwd, stp, mmd, thrp, mat_found=mat_found
        )

        if rlen_new <= 0:
            break

        v_rad.insert(0, rrad_new)
        v_rad2.insert(0, rrad_new)
        v_len.insert(0, rlen_new)
        v_cx.insert(0, rx_new)
        v_cy.insert(0, ry_new)
        mm.insert(0, mmi_new)

        rx = rx_new
        ry = ry_new
        rrad = rrad_new

    # Jei neišmatavo nei vieno
    if len(v_len) == 0:
        return [], 0.0, 0.0, None

    # =====================================================================
    # 5. Kampų normalizavimas
    # Visi kampai  [0, PI] sritis, tada shift kad pirmas=PI/2

    for i in range(len(v_rad2)):
        if v_rad2[i] < 0:
            v_rad2[i] += PI2
        if v_rad2[i] > PI2:
            v_rad2[i] -= PI2
        if v_rad2[i] > PI:
            v_rad2[i] -= PI

     #shift visus
    rdiff = PIh - v_rad2[0]
    for i in range(len(v_rad2)):
        v_rad2[i] += rdiff
        if v_rad2[i] > PI:
            v_rad2[i] -= PI

    # =====================================================================
    # 6. Pirminis vidurkis ir SD
    mean = sum(v_len) / len(v_len)
    rmean = sum(v_rad2) / len(v_rad2)

    var = sum((v - mean) ** 2 for v in v_len) / len(v_len)
    rvar = sum((v - rmean) ** 2 for v in v_rad2) / len(v_rad2)

    sd = math.sqrt(var)
    rsd = math.sqrt(rvar)

    # =====================================================================
    # 7. avgVector versijos
    v_len_avg = avgVector(v_len, 1, 2)
    v_cx_avg = avgVector(v_cx, 1, 2)
    v_cy_avg = avgVector(v_cy, 1, 2)
    v_rad_avg = avgVector(v_rad, 1, 2)
    v_rad2_avg = avgVector(v_rad2, 1, 2)

    # =====================================================================
    # 8. Outlier filtravimas
    for i in range(len(v_len) - 1, -1, -1):
        if v_len[i] > mean + 1.0 * sd or v_len[i] < mean - 1.0 * sd:
            # Pakeisti suvidurkinomis versijomis
            v_rad[i] = v_rad_avg[i]
            v_rad2[i] = v_rad2_avg[i]
            v_len[i] = v_len_avg[i]
            v_cx[i] = v_cx_avg[i]
            v_cy[i] = v_cy_avg[i]

            nx = v_cx[i]
            ny = v_cy[i]
            rrad_i = v_rad[i]

            rx_re, ry_re, rrad_re, rlen_re, mmi_re = measureVessel7_iteration(
                nx, ny, rrad_i, rrad_i,
                img_g, img_g2, img_r, img_vsl_thn, img_mask,
                sc, pwd, stp, mmd, thrp, mat_found=mat_found
            )

            if rlen_re > 0:
                v_cx[i] = rx_re
                v_cy[i] = ry_re
                v_rad[i] = rrad_re
                v_len[i] = rlen_re
                mm[i] = mmi_re
                # Normalizuoti kampą
                v_rad2[i] = rrad_re
                if v_rad2[i] < 0:
                    v_rad2[i] += PI2
                if v_rad2[i] > PI2:
                    v_rad2[i] -= PI2
                if v_rad2[i] > PI:
                    v_rad2[i] -= PI

    # =====================================================================
    # 9. Galinis vidurkinimas
    v_len = avgVector(v_len, 1, 2)
    v_cx = avgVector(v_cx, 1, 2)
    v_cy = avgVector(v_cy, 1, 2)

    # =====================================================================
    # 10. Galinis vidurkis ir SD
    mean = sum(v_len) / len(v_len)
    rmean = sum(v_rad2) / len(v_rad2)

    var = sum((v - mean) ** 2 for v in v_len) / len(v_len)
    rvar = sum((v - rmean) ** 2 for v in v_rad2) / len(v_rad2)

    sd = math.sqrt(var)
    rsd = math.sqrt(rvar)

    # =====================================================================
    # 11. Kokybės patikra (C++ sąlyga)
    # C++ measureVesselPoint (file.h:8906):
    #   if (sd < 2.0 && v_len.size() >= nvsl_min && rsd < 2.0
    #       && mean > sc*2.0 && mean < sc*12.0)
    #
    # ankstesnis sd < 3 buvo iš KITOS funkcijos (measureVesselNew,file.h:8603), kuri nenaudoja iš measure_od_points .

    if sd >= 2.0 or len(v_len) < nvsl_min or rsd >= 2.0:
        return [], 0.0, 0.0, None
    if mean <= sc * 2.0 or mean >= sc * 12.0:
        return [], 0.0, 0.0, None

    # =====================================================================
    # 12. Rezultatas: vidurinis taškas, suvidurinti požymiai
    if not mm or not mm[0]:
        return [], 0.0, 0.0, None

    ftnum = len(mm[0])
    mean_features = [0.0] * ftnum

    for i in range(len(mm)):
        for j in range(ftnum):
            if j < len(mm[i]):
                mean_features[j] += mm[i][j]

    for j in range(ftnum):
        mean_features[j] /= len(mm)


    midi = len(v_len) // 2

    # Grąžiname: (požymiai, plotis=mean, kryptis, tarpiniai_taškai)
    # tarpiniai_taškai naudojami kontūrų piešimui
    vessel_points = {
        'v_rad': list(v_rad),
        'v_len': list(v_len),
        'v_cx': list(v_cx),
        'v_cy': list(v_cy),
        'midi_x': v_cx[midi],
        'midi_y': v_cy[midi],
    }

    return mean_features, mean, v_rad[midi], vessel_points


# =============================================================================
# MATAVIMAI APLINK OPTINĮ DISKĄ

def measure_od_points(od_x: int, od_y: int, od_r: int,
                      img_g: np.ndarray, img_g2: np.ndarray, img_r: np.ndarray,
                      img_mask: np.ndarray, img_vsl_thn: np.ndarray,
                      sc: float, odr_mult: List[float] = [2.0, 2.5]) -> List[dict]:
    """
    Atlieka matavimus tam tikru atstumu nuo optinio disko centro.

    Naudoja ADAPTYVIĄJĮ ciklą - radus kraujagyslę peršoka per ją.
    Naudoja mat_found kaukę.
      - Storos ribos ties 1.5 ir 3.0 × od_r → stabdo matavimus ties ribomis
      - Po kiekvieno matavimo piešiami kontūrai → neleidžia pakartotinai matuoti

    Args:
        od_x, od_y, od_r: Optinio disko koordinatės ir spindulys
        img_g: Žalias kanalas (požymių išgavimui)
        img_g2: Žalias kanalas su centrinio reflekso panaikinimu (profilio analizei)
        img_r: Raudonas kanalas
        img_mask: Akies dugno kaukė
        img_vsl_thn: Suplonintas kraujagyslių paveikslėlis
        sc: Skalės koeficientas
        odr_mult: Atstumo nuo OD dauginimo koeficientai

    Returns:
        Matavimų sąrašas
    """
    measurements = []

    pwd = 12.0 * sc
    stp = 2 * pwd / 100
    mmd = 15
    thrp = 0.5

    pd1 = 5.0   # Atstumas tarp taškų ant apskritimo pikseliais
    pd2 = 0.5   # Daugiklis kiek žingsnių peršokti radus kraujagyslę

    # Storos ribos ties 1.5 ir 3.0 × od_r neleidžia matavimams kirsti zonos
    # Po matavimo piešiami kontūrai — neleidžia pakartotinai matuoti tos pačios kraujagyslės
    h, w = img_g.shape[:2]

    mat_found = np.zeros((h, w), dtype=np.uint8)
    border_thickness = max(1, int(sc * 5.0))
    cv2.circle(mat_found, (od_x, od_y), int(1.5 * od_r), 255, border_thickness)
    cv2.circle(mat_found, (od_x, od_y), int(3.0 * od_r), 255, border_thickness)

    for mult in odr_mult:
        radius = od_r * mult
        da = pd1 / radius
        ca = da

        while ca < PI2:
            xr = round(od_x + radius * math.cos(ca))
            yr = round(od_y + radius * math.sin(ca))
            ix, iy = int(xr), int(yr)

            # Tikrinti ar taškas neužblokuotas mat_found kaukėje
            if 0 <= ix < w and 0 <= iy < h and mat_found[iy, ix] > 0:
                ca += da
                continue

            # Matuoti tašką
            features, width, vessel_angle, vessel_points = measureVesselPoint(
                xr, yr, img_g, img_g2, img_r, img_mask, img_vsl_thn,
                sc, pwd, stp, mmd, thrp, mat_found=mat_found
            )

            if features and width > 0:
                midi_x = float(xr)
                midi_y = float(yr)
                if vessel_points is not None:
                    midi_x = vessel_points.get('midi_x', float(xr))
                    midi_y = vessel_points.get('midi_y', float(yr))

                measurement = {
                    'x': midi_x, 'y': midi_y, 'angle': ca,
                    'vessel_angle': vessel_angle, 'radius': radius,
                    'width': width, 'features': features,
                    'vessel_points': vessel_points,
                }
                measurements.append(measurement)

                # Piešti kontūrus ant mat_found — neleidžia pakartotinai matuoti
                if vessel_points is not None:
                    vp = vessel_points
                    v_rad_p = vp['v_rad']
                    v_len_p = vp['v_len']
                    v_cx_p = vp['v_cx']
                    v_cy_p = vp['v_cy']
                    contours = []
                    for j in range(1, len(v_len_p)):
                        pf_x = int(v_cx_p[j] + v_len_p[j] * 0.5 * math.cos(v_rad_p[j]))
                        pf_y = int(v_cy_p[j] + v_len_p[j] * 0.5 * math.sin(v_rad_p[j]))
                        pt_x = int(v_cx_p[j] + v_len_p[j] * 0.5 * math.cos(v_rad_p[j] + PI))
                        pt_y = int(v_cy_p[j] + v_len_p[j] * 0.5 * math.sin(v_rad_p[j] + PI))
                        pfl_x = int(v_cx_p[j-1] + v_len_p[j-1] * 0.5 * math.cos(v_rad_p[j-1]))
                        pfl_y = int(v_cy_p[j-1] + v_len_p[j-1] * 0.5 * math.sin(v_rad_p[j-1]))
                        ptl_x = int(v_cx_p[j-1] + v_len_p[j-1] * 0.5 * math.cos(v_rad_p[j-1] + PI))
                        ptl_y = int(v_cy_p[j-1] + v_len_p[j-1] * 0.5 * math.sin(v_rad_p[j-1] + PI))
                        contour = np.array([
                            [pf_x, pf_y], [pt_x, pt_y],
                            [ptl_x, ptl_y], [pfl_x, pfl_y]
                        ], dtype=np.int32)
                        contours.append(contour)
                    if contours:
                        cv2.drawContours(mat_found, contours, -1, 255, -1)

                ca += pd2 * width * da / pd1

            ca += da

    return measurements


# =============================================================================
# KLASIFIKACIJA

def _stable_kmeans_av(points: np.ndarray, widths: List[float], dims: int,
                       n_seeds: int = 5) -> Tuple[np.ndarray, int]:
    """
    Stabilus k-means su voting ir width tiebreaker A/V priskyrimui.
    Grąžina (final_labels, artery_cluster_idx) kur artery_cluster_idx yra 0 arba 1.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    n = len(points)
    vote_labels = np.zeros((n_seeds, n), dtype=np.int32)
    vote_art = np.zeros(n_seeds, dtype=np.int32)

    for s in range(n_seeds):
        cv2.setRNGSeed(42 + s * 17)
        _, labels, centers = cv2.kmeans(points, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels_flat = labels.flatten()
        sum0 = sum(centers[0, k] for k in range(dims))
        sum1 = sum(centers[1, k] for k in range(dims))

        if abs(sum0 - sum1) < 0.1:
            w0 = np.mean([widths[i] for i in range(n) if labels_flat[i] == 0]) if np.any(labels_flat == 0) else 0
            w1 = np.mean([widths[i] for i in range(n) if labels_flat[i] == 1]) if np.any(labels_flat == 1) else 0
            art_cluster = 0 if w0 < w1 else 1
        else:
            art_cluster = 0 if sum0 > sum1 else 1

        vote_labels[s] = labels_flat
        vote_art[s] = art_cluster

    canonical = np.where(vote_art[:, None] == vote_labels, 1, 0)
    final_art_mask = (canonical.sum(axis=0) > n_seeds / 2).astype(np.int32)
    most_common_art_cluster = int(np.argmax(np.bincount(vote_art)))
    final_labels = np.where(final_art_mask == 1, most_common_art_cluster, 1 - most_common_art_cluster)

    return final_labels, most_common_art_cluster


def classifyVessels(vslsave: List[List[float]], od_x: int, od_y: int,
                    od_r: int = 100,
                    class_sel: List[int] = CLASS_SEL,
                    segment_counts: List[int] = None,
                    norm_method: str = 'spatial',
                    kmeans_method: str = 'stable') -> dict:
    """
    Klasifikuoja kraujagysles į arterijas ir venas.

    Naudoja k-means klasterizavimą su atrinktais požymiais.
    Klasės:
    1 - Viršutinė arterija
    2 - Viršutinė vena
    3 - Apatinė arterija
    4 - Apatinė vena

    Args:
        vslsave: Kraujagyslių matavimų duomenys
        od_x, od_y: Optinio disko koordinatės
        class_sel: Požymių indeksai klasifikacijai

    Returns:
        Klasifikacijos rezultatai
    """
    if not vslsave:
        return {'classes': [], 'stats': {}}

    dims = len(class_sel)

    # ištraukti koordinates ir požymius
    crdx = []
    crdy = []
    vsl_wd = []
    vsl_vals = []
    seg_counts = []  # segmentų skaičius kiekvienam matavimui
    orig_indices = []  # originalūs indeksai vslsave masyve

    for i in range(len(vslsave)):
        # Pirmiausia patikrinti ar yra visi požymiai
        vsl_vals_pack = []
        for k in range(dims):
            if class_sel[k] < len(vslsave[i]):
                vsl_vals_pack.append(vslsave[i][class_sel[k]])

        # Tik jei visi požymiai yra, pridėti matavimą
        if len(vsl_vals_pack) == dims:
            crdx.append(vslsave[i][1])
            crdy.append(vslsave[i][2])
            vsl_wd.append(vslsave[i][4])
            vsl_vals.append(vsl_vals_pack)
            sc_val = segment_counts[i] if segment_counts and i < len(segment_counts) else 999
            seg_counts.append(sc_val)
            orig_indices.append(i)

    if not vsl_vals or len(vsl_wd) < 2:
        return {'classes': [], 'stats': {}}

    # atstumų matrica
    sz = len(crdx)
    distm = np.zeros((sz, sz), dtype=np.float32)
    for i in range(sz):
        for j in range(sz):
            dist = math.sqrt((crdx[i] - crdx[j])**2 + (crdy[i] - crdy[j])**2)
            distm[i, j] = dist

    nn = 10
    if nn > sz:
        nn = sz

    points = np.zeros((sz, dims), dtype=np.float32)

    # atskirti viršutinius ir apatinius
    num_top = 0
    num_btm = 0
    tb_idx = []
    tt_idx = []
    tbt_idt = []

    for i in range(sz):
        tb_idx.append(num_btm)
        tt_idx.append(num_top)

        if crdy[i] > od_y:
            tbt_idt.append(1)
            num_btm += 1
        else:
            tbt_idt.append(0)
            num_top += 1

    points_btm = np.ones((max(num_btm, 1), dims), dtype=np.float32) if num_btm > 0 else None
    points_top = np.ones((max(num_top, 1), dims), dtype=np.float32) if num_top > 0 else None

    # =========================================================================
    # NORMALIZAVIMAS
    if norm_method == 'zscore':
        # Paprastas z-score
        arr = np.array(vsl_vals, dtype=np.float64)
        means = arr.mean(axis=0)
        sds = arr.std(axis=0)
        sds[sds == 0] = 1.0

        for i in range(sz):
            for k in range(dims):
                val = (vsl_vals[i][k] - means[k]) / sds[k]
                points[i, k] = val
                if crdy[i] > od_y:
                    if points_btm is not None:
                        points_btm[tb_idx[i], k] = val
                else:
                    if points_top is not None:
                        points_top[tt_idx[i], k] = val

    elif norm_method == 'spatial_noself':
        for i in range(sz):
            nni = getNN(distm, i, nn, sz)
            vsl_sums = [0.0] * dims
            vsl_means = [0.0] * dims
            vsl_sds = [0.0] * dims
            sumds = 0.0
            for j in range(len(nni)):
                if nni[j] == i:
                    continue
                crd = distm[i, nni[j]]
                if crd == 0:
                    crd = 1
                sumds += 1.0 / crd
                for k in range(dims):
                    vsl_sums[k] += vsl_vals[nni[j]][k] / crd
            for k in range(dims):
                vsl_means[k] = vsl_sums[k] / sumds if sumds > 0 else 0
            for j in range(len(nni)):
                if nni[j] == i:
                    continue
                crd = distm[i, nni[j]]
                if crd == 0:
                    crd = 1
                for k in range(dims):
                    vsl_sds[k] += (vsl_vals[nni[j]][k] / crd - vsl_means[k])**2
            for k in range(dims):
                vsl_sds[k] = math.sqrt(vsl_sds[k] / sumds) if sumds > 0 else 1.0
            for k in range(dims):
                if vsl_sds[k] > 0:
                    val = (vsl_vals[i][k] - vsl_means[k]) / vsl_sds[k]
                else:
                    val = 0.0
                points[i, k] = val
                if crdy[i] > od_y:
                    if points_btm is not None:
                        points_btm[tb_idx[i], k] = val
                else:
                    if points_top is not None:
                        points_top[tt_idx[i], k] = val

    else:  # 'spatial' — originalus erdvinis (formulės 24-26)
        for i in range(sz):
            nni = getNN(distm, i, nn, sz)
            vsl_sums = [0.0] * dims
            vsl_means = [0.0] * dims
            vsl_sds = [0.0] * dims
            sumds = 0.0
            for j in range(len(nni)):
                crd = distm[i, nni[j]]
                if crd == 0:
                    crd = 1
                sumds += 1.0 / crd
                for k in range(dims):
                    vsl_sums[k] += vsl_vals[nni[j]][k] / crd
            for k in range(dims):
                vsl_means[k] = vsl_sums[k] / sumds if sumds > 0 else 0
            for j in range(len(nni)):
                crd = distm[i, nni[j]]
                if crd == 0:
                    crd = 1
                for k in range(dims):
                    vsl_sds[k] += (vsl_vals[nni[j]][k] / crd - vsl_means[k])**2
            for k in range(dims):
                vsl_sds[k] = math.sqrt(vsl_sds[k] / sumds) if sumds > 0 else 1.0
            for k in range(dims):
                if vsl_sds[k] > 0:
                    val = (vsl_vals[i][k] - vsl_means[k]) / vsl_sds[k]
                else:
                    val = 0.0
                points[i, k] = val
                if crdy[i] > od_y:
                    if points_btm is not None:
                        points_btm[tb_idx[i], k] = val
                else:
                    if points_top is not None:
                        points_top[tt_idx[i], k] = val

    # =========================================================================
    #  K-MEANS KLASIFIKACIJA
    vsl_class = []

    if kmeans_method == 'threshold_m1':
        m1_feat_idx = class_sel.index(5) if 5 in class_sel else 0
        for i in range(sz):
            val = points[i, m1_feat_idx]
            measure_type = 1 if val > 0 else 2
            if crdy[i] > od_y:
                measure_type += 2
            vsl_class.append(measure_type)

    elif kmeans_method == 'threshold_m1_median':
        m1_feat_idx = class_sel.index(5) if 5 in class_sel else 0
        top_vals = [points[i, m1_feat_idx] for i in range(sz) if crdy[i] <= od_y]
        btm_vals = [points[i, m1_feat_idx] for i in range(sz) if crdy[i] > od_y]
        top_thr = float(np.median(top_vals)) if len(top_vals) > 0 else 0.0
        btm_thr = float(np.median(btm_vals)) if len(btm_vals) > 0 else 0.0
        for i in range(sz):
            val = points[i, m1_feat_idx]
            if crdy[i] > od_y:
                measure_type = 3 if val > btm_thr else 4
            else:
                measure_type = 1 if val > top_thr else 2
            vsl_class.append(measure_type)

    elif len(vsl_wd) > 1:
        # jei tik viena pusė ARBA per mažai taškų k-means
        # K-means su k=2 reikalauja bent 2 taškų kiekvienoje grupėje
        if num_top < 2 or num_btm < 2:
            #  bendras k-means visiems taškams
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

            # Patikrinti ar turime bent 2 taškus iš viso
            if len(points) < 2:
                vsl_class = [1 if crdy[i] <= od_y else 3 for i in range(sz)]
            else:
                if kmeans_method == 'stable':
                    labels_flat, art_cluster = _stable_kmeans_av(points, vsl_wd, dims)
                    for i in range(sz):
                        measure_type = 1 if labels_flat[i] == art_cluster else 2
                        if crdy[i] > od_y:
                            measure_type += 2
                        vsl_class.append(measure_type)
                else:
                    _, labels, centers = cv2.kmeans(points, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                    sum1 = sum(centers[0, k] for k in range(dims))
                    sum2 = sum(centers[1, k] for k in range(dims))

                    for i in range(sz):
                        clusterIdx = int(labels[i, 0])

                        measure_type = 1
                        if (clusterIdx == 0 and sum1 < sum2) or (clusterIdx == 1 and sum2 <= sum1):
                            measure_type = 2

                        if crdy[i] > od_y:
                            measure_type += 2

                        vsl_class.append(measure_type)
        else:
            try:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

                if kmeans_method == 'stable':
                    wd_top = [vsl_wd[i] for i in range(sz) if tbt_idt[i] == 0]
                    wd_btm = [vsl_wd[i] for i in range(sz) if tbt_idt[i] == 1]
                    labels_top_flat, art_top = _stable_kmeans_av(points_top[:num_top], wd_top, dims)
                    labels_btm_flat, art_btm = _stable_kmeans_av(points_btm[:num_btm], wd_btm, dims)

                    for i in range(sz):
                        if tbt_idt[i] == 0:
                            measure_type = 1 if labels_top_flat[tt_idx[i]] == art_top else 2
                        else:
                            measure_type = 3 if labels_btm_flat[tb_idx[i]] == art_btm else 4
                        vsl_class.append(measure_type)
                else:
                    _, labels_top, centers_top = cv2.kmeans(points_top[:num_top], 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    _, labels_btm, centers_btm = cv2.kmeans(points_btm[:num_btm], 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                    if centers_top.shape[1] != dims or centers_btm.shape[1] != dims:
                        raise ValueError(f"Invalid centers shape: top={centers_top.shape}, btm={centers_btm.shape}, expected dims={dims}")

                    sum1t = sum(centers_top[0, k] for k in range(dims))
                    sum2t = sum(centers_top[1, k] for k in range(dims))
                    sum1b = sum(centers_btm[0, k] for k in range(dims))
                    sum2b = sum(centers_btm[1, k] for k in range(dims))

                    for i in range(sz):
                        if tbt_idt[i] == 0:
                            measure_type = 1
                            clusterIdx = int(labels_top[tt_idx[i], 0])
                            if (clusterIdx == 0 and sum1t < sum2t) or (clusterIdx == 1 and sum2t <= sum1t):
                                measure_type = 2
                        else:
                            measure_type = 3
                            clusterIdx = int(labels_btm[tb_idx[i], 0])
                            if (clusterIdx == 0 and sum1b < sum2b) or (clusterIdx == 1 and sum2b <= sum1b):
                                measure_type = 4

                        vsl_class.append(measure_type)

            except Exception as e:
                vsl_class = [1 if crdy[i] <= od_y else 3 for i in range(sz)]

    # Jei klasifikacija nepavyko, priskirti numatytąsias klases
    if not vsl_class:
        vsl_class = [1 if crdy[i] <= od_y else 3 for i in range(sz)]

    # =========================================================================
    # PARINKTI PLAČIAUSIĄ KRAUJAGYSLĘ KIEKVIENOJE KLASĖJE

    artery_tm = -1  # Viršutinė arterija (klasė 1)
    vein_tm = -1    # Viršutinė vena (klasė 2)
    artery_bm = -1  # Apatinė arterija (klasė 3)
    vein_bm = -1    # Apatinė vena (klasė 4)

    # Surinkti VISUS pločius pagal klasę
    widths_by_class = {1: [], 2: [], 3: [], 4: []}

    for i in range(len(vsl_class)):
        width = vsl_wd[i] if i < len(vsl_wd) else 0
        cls = vsl_class[i]
        if cls in widths_by_class:
            widths_by_class[cls].append(width)

        if vsl_class[i] == 1:
            if artery_tm < 0 or width > vsl_wd[artery_tm]:
                artery_tm = i
        elif vsl_class[i] == 2:
            if vein_tm < 0 or width > vsl_wd[vein_tm]:
                vein_tm = i
        elif vsl_class[i] == 3:
            if artery_bm < 0 or width > vsl_wd[artery_bm]:
                artery_bm = i
        elif vsl_class[i] == 4:
            if vein_bm < 0 or width > vsl_wd[vein_bm]:
                vein_bm = i

    # Statistika — VISŲ klasifikuotų kraujagyslių vidurkiai ir SD
    stats = {
        'mean': [0.0, 0.0, 0.0, 0.0],
        'sd': [0.0, 0.0, 0.0, 0.0],
        'count': [0, 0, 0, 0],
        'widest': [0.0, 0.0, 0.0, 0.0],  # Plačiausios
        'main_indices': {
            'artery_top': artery_tm,
            'vein_top': vein_tm,
            'artery_bottom': artery_bm,
            'vein_bottom': vein_bm
        },
        'main_coords': {
            'artery_top': (crdx[artery_tm], crdy[artery_tm]) if artery_tm >= 0 else None,
            'vein_top': (crdx[vein_tm], crdy[vein_tm]) if vein_tm >= 0 else None,
            'artery_bottom': (crdx[artery_bm], crdy[artery_bm]) if artery_bm >= 0 else None,
            'vein_bottom': (crdx[vein_bm], crdy[vein_bm]) if vein_bm >= 0 else None
        }
    }

    # Užpildyti visų kraujagyslių vidurkius ir SD
    for cls_idx, cls_key in enumerate([1, 2, 3, 4]):
        w_list = widths_by_class[cls_key]
        stats['count'][cls_idx] = len(w_list)
        if w_list:
            stats['mean'][cls_idx] = sum(w_list) / len(w_list)
            if len(w_list) > 1:
                mean = stats['mean'][cls_idx]
                stats['sd'][cls_idx] = math.sqrt(
                    sum((w - mean) ** 2 for w in w_list) / len(w_list)
                )

    # Plačiausios kraujagyslės plotis
    if artery_tm >= 0:
        stats['widest'][0] = vsl_wd[artery_tm]
    if vein_tm >= 0:
        stats['widest'][1] = vsl_wd[vein_tm]
    if artery_bm >= 0:
        stats['widest'][2] = vsl_wd[artery_bm]
    if vein_bm >= 0:
        stats['widest'][3] = vsl_wd[vein_bm]

    return {
        'classes': vsl_class,
        'stats': stats
    }


# =============================================================================
# REZULTATŲ SKAIČIAVIMAS


def calculate_avr(stats: dict) -> dict:
    """
    Apskaičiuoja AVR (Arteriolar-to-Venular Ratio).
    Naudoja plačiausią kraujagyslę kiekvienoje klasėje.

    Args:
        stats: Klasifikacijos statistika

    Returns:
        AVR rezultatai
    """
    avr = {
        'top': 0.0,
        'bottom': 0.0,
        'total': 0.0
    }

    # Naudoti plačiausias kraujagysles (C++ pick-widest), fallback į mean
    widths = stats.get('widest', stats.get('mean', [0, 0, 0, 0]))

    # Viršutinis AVR (arterija / vena)
    if widths[1] > 0:  # Vena
        avr['top'] = widths[0] / widths[1]

    # Apatinis AVR
    if widths[3] > 0:  # Vena
        avr['bottom'] = widths[2] / widths[3]

    # Bendras AVR
    artery_mean = (widths[0] + widths[2]) / 2
    vein_mean = (widths[1] + widths[3]) / 2

    if vein_mean > 0:
        avr['total'] = artery_mean / vein_mean

    return avr