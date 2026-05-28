# REALM-VPCE

**REALM** (Robotic Environment for Autonomous Learning and Mapping) extended 
with the **Visual Place Cell Encoding (VPCE)** model — a biologically inspired 
framework for generating spatially structured, place-cell-like activation 
patterns from robot-acquired visual input using unsupervised feature clustering 
and radial basis function encoding.

This repository is a fork of [REALM](https://github.com/CJHRobotics/REALM) 
developed at the USF BioRobotics Lab. It extends the base simulation 
infrastructure with the VPCE model and serves as the active codebase for 
ongoing revisions targeting publication in a peer-reviewed journal.

---

## What is VPCE?

VPCE models hippocampal place-cell-like representations without relying on 
odometry, path integration, ground-truth coordinates, or task feedback. The 
model processes point-of-view images collected during robot exploration, 
extracts high-dimensional visual features, and clusters them in feature space 
to define a population of visual place cells. Each place cell is characterized 
by the center and spread of a cluster; activation is computed via a radial 
basis function over the distance between a new observation and the stored 
cluster centroids.

### Pipeline

1. **Feature Extraction** — Each image is passed through a pretrained ResNet50 
   combined with handcrafted descriptors (HOG, color histograms, spatial 
   histograms) to produce a multimodal feature vector.
2. **Ensemble Formation** — Feature vectors are clustered using k-means or GMM. 
   Each cluster centroid defines a visual place cell and its receptive field 
   in feature space.
3. **Activation Encoding** — New observations are encoded as a graded 
   activation pattern across the ensemble using RBF scoring against stored 
   centroids.

### Key Properties Evaluated

- Spatial proximity encoding
- Boundary and wall separation (statistically validated)
- Local remapping under structural change
- Population sparseness and spatial information content
- Multi-field place cell emergence under GMM clustering (preliminary)

---

## Repository Status

This repository tracks active revisions to the VPCE model following peer 
review. Current development priorities include:

- Reframing the biological motivation around primate spatial view cells
- Explicit repositioning of the RBF assumption as a modeling design choice
- Addition of statistical significance tests across all evaluation metrics
- Conversion of tabular results to distributional figures
- Investigation of sparsity mechanisms (thresholding, k-winners-take-all)
- Integration of VPCE as a state representation module for RL agents

---

## Requirements

### 1. Python 3.11

**macOS:**
```bash
brew install python@3.11
```

**Linux:**
```bash
sudo apt-get install python3.11
```

**Windows:** Install Python 3.11 from the 
[Microsoft Store](https://apps.microsoft.com/store/detail/python-311/9NRWMJP3717K) 
to avoid PATH issues.

### 2. Webots R2025a

Download and install from the [Cyberbotics website](https://cyberbotics.com/#download).

> **Linux users:** Do not install Webots via Snap. Use the `.deb` package or 
> tarball instead.

### 3. Git

**Windows:** [git-scm.com](https://git-scm.com/download/win)  
**Linux:** `sudo apt-get install git`  
**macOS:** `brew install git`

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd REALM-VPCE
```

### 2. Run the install script

```bash
# Core environment only (Webots + RL training)
python setup/realm_install.py

# Core + analysis environment (adds Jupyter, statsmodels, seaborn, etc.)
python setup/realm_install.py --data_analysis
```

This will:
- Find Python 3.11 on your system
- Create `realm_core_venv` (and optionally `realm_analysis_venv`)
- Install all dependencies
- Add the project root to the venv's Python path
- Generate `runtime.ini` files in all Webots controller directories

To remove the environments:
```bash
python setup/realm_install.py --uninstall
python setup/realm_install.py --uninstall --data_analysis
```

### 3. Activate the environment

**macOS/Linux:**
```bash
source realm_core_venv/bin/activate
```

**Windows:**
```bash
realm_core_venv\Scripts\activate
```

For analysis work:
```bash
source realm_analysis_venv/bin/activate   # macOS/Linux
realm_analysis_venv\Scripts\activate      # Windows
```

> If you add a new controller under `simulation/controllers/`, re-run 
> `python setup/add_runtime_ini.py` to generate its `runtime.ini`.

---

## Project Structure

```
REALM-VPCE/
├── setup/
│   ├── realm_install.py          # Install / uninstall script
│   ├── add_runtime_ini.py        # Generates Webots runtime.ini files
│   ├── requirements_webots.txt   # Core venv dependencies
│   └── requirements_analysis.txt # Analysis venv dependencies
│
├── realm_tools/
│   ├── robot_lib/
│   │   ├── hambot.py             # Base robot class (sensors, motors, supervisor)
│   │   ├── my_robot.py           # User extension template (inherits HamBot)
│   │   └── robot_tools.py        # Shared robot utility functions
│   ├── simulation_lib/
│   │   ├── environment.py        # Environment class and environment objects
│   │   ├── maze_parser.py        # XML maze file parser
│   │   └── webots_torch_environment.py  # Gymnasium environment skeleton
│   └── image_lib/
│       ├── feature_extractor.py  # CNN feature extraction (ResNet50 + HOG)
│       └── image_feature_lib.py  # Image processing utilities
│
├── simulation/
│   ├── controllers/
│   │   ├── example/              # Example Webots controller
│   │   └── calibration/          # Keyboard-driven calibration controller
│   ├── protos/                   # HamBot and world object Webots protos
│   └── worlds/                   # Webots world files and maze XMLs
│
├── data/
│   └── DataCache/                # Temp files used by the display system
│
└── docs/                         # Figures and documentation assets
```

---

## Extending for Your Project

The intended workflow when forking this repo for a new experiment:

1. **Robot logic** — subclass `HamBot` in `my_robot.py` and add 
   experiment-specific methods (action sets, observation processing, etc.)
2. **Environment** — fill in the `WebotsEnv` skeleton in 
   `webots_torch_environment.py` with your observation space, reward 
   function, and episode logic
3. **Controllers** — add new Webots controllers under 
   `simulation/controllers/` then re-run `add_runtime_ini.py`
4. **Personal files** — your personal robot subclass (e.g. 
   `yourname_robot.py`) can be gitignored so the template stays clean 
   for others

---

## Calibration Controller

A keyboard-driven controller is provided for manually testing robot 
behaviour in Webots:

| Key           | Action      |
|---------------|-------------|
| Arrow Up      | Forward     |
| Arrow Down    | Backward    |
| Arrow Left    | Turn left   |
| Arrow Right   | Turn right  |
| Any other key | Stop        |

Open `simulation/worlds/calibration.wbt` in Webots to use it.

---

## Additional Documentation

- [HamBot Reference](realm_tools/README.md)
- [Simulation & Webots Guide](simulation/README.md)
- [Controller Setup Guide](simulation/controllers/README.md)
- [Webots Controller Guide](https://cyberbotics.com/doc/guide/controller-programming)
- [Gymnasium API Reference](https://gymnasium.farama.org/api/env/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

## Funding

This work was supported in part by NSF IIS Robust Intelligence grant 
#1703225 — *Experimental and Robotics Investigations of Multiscale Spatial 
Memory Consolidation in Complex Environments*, University of South Florida.