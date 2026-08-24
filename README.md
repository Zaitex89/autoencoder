# Autoencoder på MNIST

Ett litet maskininlärningsprojekt som tränar en **autoencoder** att komprimera
och återskapa handskrivna siffror från MNIST-datasetet. Modellen pressar ihop
varje bild (784 pixlar) till en liten latent kod (32 tal) och bygger sedan upp
bilden igen — helt utan etiketter.

Projektet består av tre steg som körs i tur och ordning:

**main.py** (träna) → **eda.py** (utforska datan) → **evaluate.py** (utvärdera)

---

## Krav

- Python 3.9 eller senare
- Följande paket:

```
pip install -r requirements.txt
```

`scikit-image` behövs bara för `evaluate.py` (det används för SSIM-måttet).

### Virtuell miljö (rekommenderas)

Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Projektstruktur

```
autoencoder/
├── main.py            # Tränar modellen och sparar vikterna
├── eda.py             # Utforskande dataanalys (EDA) av MNIST
├── evaluate.py        # Laddar den tränade modellen och mäter kvaliteten
├── autoencoder.pth    # Skapas av main.py (modellens vikter)
└── data/              # Skapas automatiskt – MNIST laddas ner hit
```

---

## Så här kör du

Kör filerna i den här ordningen från projektmappen.

### 1. `main.py` — träna modellen

```
python main.py
```

Laddar ner MNIST första gången (till `data/`), tränar autoencodern i 20 epoker
och sparar de färdiga vikterna.

Skapar:
- `autoencoder.pth` — modellens vikter (behövs av steg 3)
- `reconstructions.png` — original överst, återskapade siffror underst

### 2. `eda.py` — utforska datan

```
python eda.py
```

Skriver ut en sammanfattning i terminalen (antal bilder och klassfördelning)
och sparar två figurer.

Skapar:
- `eda_examples.png` — slumpvisa exempelbilder ur datasetet
- `eda_class_averages.png` — den genomsnittliga bilden för varje siffra

> EDA-steget är fristående och läser bara datan, så det går även bra att köra
> före `main.py` om du vill titta på datan först.

### 3. `evaluate.py` — utvärdera modellen

```
python evaluate.py
```

Laddar in `autoencoder.pth` (utan att träna om) och mäter hur väl modellen
återskapar testbilder den aldrig sett.

Skriver ut i terminalen:
- Genomsnittlig **MSE** och **MAE** (lägre = bättre)
- Genomsnittlig **SSIM** (1.0 = perfekt)

Skapar:
- `evaluation_comparison.png` — original, rekonstruktion och skillnad (felkarta)

> **Obs:** `evaluate.py` kräver att `autoencoder.pth` finns, så `main.py` måste
> ha körts först. Annars visas ett felmeddelande som påminner om detta.

---

## Filer som skapas

| Fil | Skapas av | Innehåll |
|-----|-----------|----------|
| `autoencoder.pth` | main.py | Modellens tränade vikter |
| `reconstructions.png` | main.py | Original vs. återskapade siffror |
| `eda_examples.png` | eda.py | Exempelbilder ur MNIST |
| `eda_class_averages.png` | eda.py | Genomsnittlig bild per siffra |
| `evaluation_comparison.png` | evaluate.py | Original / rekonstruktion / skillnad |

---

## Bra att veta

- **`latent_dim` måste stämma överens.** Modellen tränas och sparas med
  `latent_dim = 32`. Om du ändrar värdet i `main.py` måste du använda samma
  värde i `evaluate.py`, annars matchar inte de sparade vikterna lagren.
- **Träna en gång, utvärdera flera gånger.** När `autoencoder.pth` väl finns går
  `evaluate.py` på några sekunder eftersom den inte tränar om.
- **Editorn hittar inte `torch`?** Kontrollera att den använder rätt Python-tolk
  (din `.venv`). Det påverkar inte om koden körs, bara autocomplete och varningar.

---

## Idéer för att bygga vidare

- Minska flaskhalsen (`latent_dim = 8` eller `2`) och jämför MSE/SSIM.
- Denoising-autoencoder: lägg brus på indatan men träna mot den rena bilden.
- Convolutional autoencoder (byt `Linear` mot `Conv2d`) för skarpare bilder.