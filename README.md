# AKPMA-EF
Akies dugno  nuotraukų automatinės kraujagyslių analizės įrankis. Aptinka optinį diską, išskiria arterijas ir venas, matuoja jų plotį bei apskaičiuoja arterijų ir venų santykį (AVR).
 
Programa turi du paleidimo rėžimus:
- **GUI** - interaktyvus režimas vienos nuotraukos analizei.
- **CLI** - komandinės eilutės režimas vienai nuotraukai arba visam aplankui.

## Reikalavimai
 
- **Python:** 3.10+ (testuota su 3.10–3.12)
- **OS:** Windows, Linux, macOS
- **Bibliotekos:** žr. `requirements.txt`
  - `numpy`, `scipy`, `opencv-python`, `scikit-learn`, `scikit-image`, `Pillow`, `numba`, `pandas`, `PyQt5`

## Diegimas
 
```bash
# 1. Klonuoti repozitoriją
git clone https://github.com/laurinykas/vessel-auto-measure.git
cd vessel-auto-measure
 
# 2. Sukurti virtualią aplinką
python -m venv .venv
 
# 3. Aktyvuoti
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
 
# 4. Įdiegti priklausomybes
pip install -r requirements.txt
```

## Naudojimas
 
Visos komandos paleidžiamos iš `src/` aplanko:
 
```bash
cd src
```
 
### GUI režimas
 
```bash
python main.py
```

### CLI režimas
 
**Viena nuotrauka:**
 
```bash
python main.py --cli -i /kelias/iki/nuotraukos.jpg -o /kelias/iki/rezultatu/
```
 
**Visas aplankas:**
 
```bash
python main.py --cli -i /kelias/iki/ivesties/ -o /kelias/iki/rezultatu/
```
 
Tas pats veikia ir tiesiai per `cli.py`:
 
```bash
python cli.py -i image.jpg -o results/
python cli.py -i input_folder/ -o output_folder/
```

**Flag'ai:**
 
| Flag | Reikšmė |
|---------|---------|
| `--cli` | Paleidžia be GUI (reikalingas `main.py` atveju) |
| `-i`, `--input` | Įvesties failas arba aplankas |
| `-o`, `--output` | Išvesties aplankas (numatytasis: `output/`) |
 
**Palaikomi formatai:** `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.ppm`, `.bmp`.

## Išvestis
 
Kiekvienai įvesties nuotraukai sugeneruojami trys failai su priesaga `_rez`:
 
| Failas | Turinys |
|--------|---------|
| `{vardas}_rez.png` | Vizualizacija: originalas + OD + AVR žiedas + klasifikuotos kraujagyslės |
| `{vardas}_rez.csv` | Suvestinė: viršutinės/apatinės A ir V pločiai + AVR (viršuje/apačioje) |
| `{vardas}_rez2.csv` | Detalūs duomenys: skalė, OD koordinatės ir spindulys, visų matavimų sąrašas, klasifikacija |

## Projekto struktūra
 
```
vessel-auto-measure/
├── requirements.txt
└── src/
    ├── main.py                 GUI / CLI 
    ├── cli.py                  CLI (vienos nuotraukos / aplanko apdorojimas)
    ├── gui.py                  PyQt5 grafinė sąsaja
    ├── config.py               Parametrai, skalės formulė, duomenų struktūros
    ├── processing.py           Pagrindinis orchestratorius (VesselProcessor)
    ├── masking.py              FOV kaukės generavimas
    ├── preprocessing.py        CLAHE ir kt. vaizdo paruošimo žingsniai
    ├── vessel_extraction.py    Kraujagyslių segmentacija ir suplonimas
    ├── optic_disc.py           Optinio disko aptikimas
    └── vessel_measurement.py   Pločio matavimai, klasifikacija, AVR
```
## Pastabos
 
- Pirmas paleidimas gali užtrukti ilgiau dėl `numba` JIT kompiliacijos.
---


