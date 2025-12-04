Ce README est en français. Pour la version anglaise, voir [README_en.md](README_en.md).

# Système Unifié d’Analyse Intelligente d’Images Médicales

Classification des cellules sanguines cancéreuses (PyTorch)
Détection des tumeurs cérébrales (YOLOv8)

## Présentation du projet

Dans un laboratoire biomédical spécialisé en imagerie médicale, l’objectif est d’automatiser l’analyse de deux types critiques de pathologies :

- Les tumeurs cérébrales à partir d’IRM/scanners
- Les cancers des cellules sanguines (leucémies) à partir de frottis sanguins

Le projet consiste à développer une solution unifiée d’analyse d’images médicales basée sur le deep learning, combinant :

- Un pipeline **PyTorch** pour la classification de cellules sanguines
- Un pipeline **YOLOv8** pour la détection d’objets (tumeurs) dans des images cérébrales
- Une interface **Streamlit** permettant d’utiliser les deux modèles de façon interactive

## 1. Classification des cellules sanguines cancéreuses (PyTorch)

### Objectif

Construire un modèle basé sur `GoogLeNet` pré-entraîné pour classer différentes catégories de cellules sanguines anormales.

### Étapes du pipeline

#### 1. Chargement et vérification des images

- Charger le dataset
- Vérifier les extensions autorisées : `jpeg`, `jpg`, `bmp`, `png`
- Supprimer les fichiers invalides
- Utiliser try / except pour gérer les images corrompues

#### 2. Explorer les classes

- Chaque classe = un dossier
- Afficher le nombre d’images par classe via countplot
- Visualiser quelques images par classe

#### 3. Diviser le dataset

- Séparer les images selon :
    - **70 %** → Entraînement
    - **15 %** → Validation
    - **15 %** → Test
- Puis compter les images dans chaque dossier.

#### 4. Augmentation de données

- Sur le dataset d’entraînement :
    - blur
    - bruit (noise)
    - flip horizontal / vertical
- Objectif :
    - équilibrer les classes
    - augmenter le volume de données

#### 5. Utiliser les Transforms PyTorch

Dans `ImageFolder` :
- redimensionnement
- conversion en tenseurs
- normalisation

#### 6. DataLoader

Créer des `DataLoaders` pour :
- Charger par batch
- Mélanger les données (shuffle=True)

#### 7. Modèle

- Charger le modèle pré-entraîné GoogLeNet
- Remplacer la couche fully connected par un réseau adapté au nombre de classes du dataset

#### 8. Hyperparamètres

Définir :
- learning rate
- loss function (ex. CrossEntropyLoss)
- optimizer (ex. Adam, SGD)

#### 9. Entraînement du modèle

- Boucle d’entraînement complète
- Validation à chaque epoch
- Sauvegarde du meilleur modèle

#### 10. Évaluation

Mesurer :
- exactitude (accuracy)
- matrice de confusion
- capacité de généralisation sur le test set

#### 11. Sauvegarde

Enregistrer :
- le modèle entraîné
- les paramètres
- la normalisation

## 2. Détection des tumeurs cérébrales (YOLOv8)

### Objectif

Classer et localiser les tumeurs sur des images d’IRM/scanners à l’aide de `YOLOv8`.

### Étapes du pipeline

#### 1. Visualisation des images et labels

Afficher quelques images par classe avec leurs boîtes englobantes (annotations .txt).

#### 2. Préparation du dataset

Créer un dossier propre après filtrage :
    - Vérifier pour chaque image la présence d’un label .txt
    - Si label existe → copier vers images/train, images/valid, images/test
    - Copier également les labels vers labels/train etc.
    - Si label absent → afficher un avertissement et ignorer l’image

#### 3. Fichiers de configuration YOLO

**data.yaml**

Contient :
- chemins d’accès (train / valid / test)
- nombre de classes
- noms des classes
- désactivation des augmentations

**data2.yaml**

Même contenu mais avec augmentations activées

#### 4. Vérification d’intégrité

- Vérifier que chaque image possède un label correspondant
- Supprimer toute image sans label
- Supprimer tout label sans image

#### 5. Statistiques

Compter :
- nombre d’images
- nombre de labels pour chaque split

#### 6. Entraînement YOLOv8

Définir :
- taille d’image
- batch size
- epochs
- learning rate
- modèle base (yolov8n, yolov8s…)

Lancer l’entraînement.

#### 7. Évaluation & tests

Mesurer :
- précision
- recall
- mAP
- performance en généralisation

#### 8. Sauvegarde du modèle

Exporter :
- best.pt
- last.pt

## 3. Interface Streamlit — Modèle Unifié

Une interface `Streamlit` permet :

- d'importer une image
- d’exécuter :
    - la classification des cellules sanguines `(PyTorch)`
    - la détection de tumeurs cérébrales `(YOLOv8)`
- d’afficher :
    - la classe prédite
    - l’image annotée par YOLO
    - les probabilités et informations du modèle

## Structure du projet

```
📁 Diagnostic-multimodal
│
├── pytorch_model/
│   ├── train.py
│   ├── data/
│   ├── saved_model.pth
│   └── utils.py
│
├── yolo_model/
│   ├── images/
│   ├── labels/
│   ├── data.yaml
│   ├── data2.yaml
│   └── runs/
│
├── streamlit_app/
│   └── app.py
│
├── notebooks/
│   ├── classification_cells.ipynb
│   ├── yolo_preparation.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
└── README.md
```

## Instructions d’Exécution

1. Cloner le projet :  
```bash
git clone https://github.com/anass17/Diagnostic-multimodal
cd Diagnostic-multimodal
```

2. Installer les dépendances :
```Bash
pip install -r requirements.txt
```

3. Lancer l’application Streamlit :
```bash
streamlit run main.py
```

4. Ouvrir l’application dans votre navigateur:
Streamlit ouvrira automatiquement une fenêtre locale, sinon rendez-vous sur : `http://localhost:8501/`

## Conclusion

Ce projet met en œuvre une solution complète de deep learning pour l’analyse d’images médicales, combinant :

- Classification d’images de cellules sanguines
- Détection de tumeurs cérébrales
- Un tableau de bord Streamlit pour un usage clinique simplifié

Il constitue un pipeline moderne et professionnel pour l'automatisation du diagnostic médical assisté par IA.

### Interface Streamlit

![Streamlit UI 1](https://github.com/user-attachments/assets/0ee84e5b-44d8-45a8-b7cc-18ea5df7c5d4)
![Streamlit UI 2](https://github.com/user-attachments/assets/3fac894d-f312-4d18-aa71-cf393c4de206)