# Modèles de Diffusion 3D — Démo Shap-E

Démonstration pratique de génération **texte → 3D** avec [Shap-E](https://github.com/openai/shap-e) (OpenAI), Open3D et PyTorch.

> Projet universitaire : *Modèles de Diffusion 3D : Fondements, Architectures et Applications en Génération de Contenu 3D*

---

## Architecture du projet

```
3D/
├── generate_3d.py      # Script principal texte → .gif / .obj
├── visualize_obj.py    # Viewer Open3D pour les maillages .obj
├── setup.sh            # Bootstrap de l'environnement conda
├── environment.yml     # Dépendances conda (GPU)
├── requirements.txt    # Dépendances pip uniquement
└── outputs/            # Fichiers générés (ignorés par git)
```

---

## Installation

### Option A — conda (recommandé)

```bash
# GPU (CUDA 11.8)
bash setup.sh

# CPU uniquement
bash setup.sh --cpu

conda activate shap-e-demo
```

### Option B — pip seul

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/openai/shap-e.git
```

---

## Utilisation

### Générer un objet 3D depuis du texte

```bash
# GIF animé + maillage .obj (par défaut)
python generate_3d.py --prompt "a red wooden chair"

# GIF uniquement, résolution 256 px
python generate_3d.py --prompt "a ceramic mug" --no-obj --gif-size 256

# Guidance plus forte (formes plus fidèles au texte)
python generate_3d.py --prompt "a rubber duck" --guidance-scale 20.0

# Générer 4 variantes en parallèle
python generate_3d.py --prompt "a spaceship" --batch-size 4
```

Les résultats sont sauvegardés dans `outputs/` :
- `<slug>.gif` — rotation 360° rendue en NeRF
- `<slug>.obj` — maillage triangulé exportable dans Blender / MeshLab

### Visualiser un maillage .obj

```bash
python visualize_obj.py outputs/a_red_wooden_chair.obj

# Capturer un screenshot sans ouvrir la fenêtre
python visualize_obj.py outputs/a_red_wooden_chair.obj --screenshot outputs/preview.png
```

---

## Paramètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `--prompt` | — | Description textuelle de l'objet |
| `--guidance-scale` | 15.0 | Force du guidage (10–20 recommandé) |
| `--batch-size` | 1 | Nombre de variantes générées |
| `--gif-size` | 128 | Résolution des frames du GIF |
| `--output-dir` | `outputs` | Dossier de sortie |

---

## Comment ça fonctionne — vue d'ensemble

```
Texte (prompt)
      │
      ▼
  [CLIP text encoder]
      │
      ▼
  Diffusion (text300M)          ← 64 étapes de débruitage
  ─ espace latent 3D ─
      │
      ▼
  Transmitter (xm)              ← décodeur latent → représentation 3D
      │
      ├──► NeRF rendering  ──► GIF animé
      │
      └──► Marching cubes ──► Maillage .obj
```

Shap-E apprend une représentation latente d'objets 3D conditionnée sur du texte ou des images. Le modèle de diffusion opère directement dans cet espace latent compact, rendant la génération beaucoup plus rapide qu'une diffusion voxel ou point-cloud brute.

---

## Configuration matérielle recommandée

| | Minimum | Recommandé |
|---|---|---|
| GPU VRAM | 4 Go | 8 Go+ |
| RAM | 8 Go | 16 Go |
| Temps / sample | ~2 min (CPU) | ~20 s (GPU) |

---

## Stack technique

- **[Shap-E](https://github.com/openai/shap-e)** — modèle de diffusion 3D d'OpenAI
- **[PyTorch](https://pytorch.org/)** — framework deep learning
- **[Open3D](http://www.open3d.org/)** — visualisation et traitement de géométrie 3D
- **[imageio](https://imageio.readthedocs.io/)** — export GIF/vidéo
- **[trimesh](https://trimsh.org/)** — manipulation de maillages

---

## Références

- Jun et al., *Shap-E: Generating Conditional 3D Implicit Functions*, 2023 — [arXiv:2305.02463](https://arxiv.org/abs/2305.02463)
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields*, ECCV 2020
