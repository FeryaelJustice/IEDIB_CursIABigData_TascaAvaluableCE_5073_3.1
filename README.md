# Classificació d'espècies de pingüí de Palmer

Aquest projecte implementa quatre classificadors diferents per predir l'espècie de pingüí a partir de les mesures dels individus: regressió logística, màquines de suport vectorial, arbres de decisió i K veïns més propers.  Els models s'entrenen amb el conjunt de dades *Palmer Penguins* i s'exposen posteriorment mitjançant una API web basada en Flask.  També s'inclou un client senzill per provar l'API.

## Estructura del projecte

```txt
penguins_project/
├── environment.yml      # definició de l'entorn Conda
├── models/              # fitxers pickle dels models entrenats
├── scripts/             # scripts de preprocés i entrenament
│   ├── utils.py         # funcions compartides per carregar i preprocesar les dades
│   ├── train_logreg.py  # entrenament de la regressió logística
│   ├── train_svm.py     # entrenament de l'SVM
│   ├── train_tree.py    # entrenament de l'arbre de decisió
│   └── train_knn.py     # entrenament del KNN
├── app/
│   └── app.py          # servidor Flask amb quatre endpoints de predicció
├── client/
│   └── client.py       # client que envia peticions a l'API
└── README.md            # aquest document
```

## Entrenament

Els scripts d'entrenament es troben a la carpeta `scripts/`.  Cada script carrega el conjunt de dades, elimina les files amb valors buits, aplica una normalització estàndard a les columnes numèriques, codifica les variables categòriques mitjançant *one‑hot encoding* i entrena un model diferent.  En finalitzar, desarà el vectoritzador, els paràmetres de l'escalat i el model entrenat al directori `models/`.

Per exemple, per entrenar la regressió logística:

```bash
conda activate penguins-env
python scripts/train_logreg.py
```

Repeteix l'operació amb els altres scripts (`train_svm.py`, `train_tree.py`, `train_knn.py`) per generar tots els models.

## API Flask

L'aplicació web es troba a `app/app.py`.  Aquesta carrega els models i exposa quatre endpoints de predicció:

- `POST /predict/logreg`
- `POST /predict/svm`
- `POST /predict/tree`
- `POST /predict/knn`

Cada petició ha d'enviar un JSON amb les característiques següents:

```json
{
  "island": "Biscoe",
  "sex": "Male",
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0
}
```

La resposta conté el codi de classe, el nom de l'espècie i les probabilitats per a cada classe.

Per posar en marxa l'API:

```bash
conda activate penguins-env
python app/app.py
```

## Client

Per provar els endpoints de predicció s'inclou un client molt simple a `client/client.py`.  Aquest envia dues sol·licituds a cada model i imprimeix les respostes rebudes.

```bash
conda activate penguins-env
python client/client.py
```

## Entorn

L'entorn Conda es defineix al fitxer `environment.yml`.  Es pot crear així:

```bash
conda env create -f environment.yml
```

## Notes

- S'utilitza la llibreria `seaborn` per carregar el dataset dels pingüins, ja que ofereix els noms de columnes homogenis.  En cas de fer servir la versió de Kaggle (`penguins_size.csv`), els scripts normalitzen els noms perquè els models funcionin igual.
- El codi està pensat perquè sigui clar i fàcil d'entendre, amb comentaris que expliquen els passos principals.
