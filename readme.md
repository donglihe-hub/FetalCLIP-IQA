# Advancing Fetal Ultrasound Image Quality Assessment in Low-Resource Settings

This repository contains code for assessing fetal ultrasound image quality using the ACOUSLIC-AI 2024 blind-sweep dataset.

---

## 🔧 Installation

### Requirements

- Python 3.9 or higher

### Setup

Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## 1. Download Data

### Option A: Raw Dataset

Download the raw dataset (v1.1) from:

- [ACOUSLIC-AI dataset](https://zenodo.org/records/12697994)

Place the downloaded zip file in the project root directory, then run:

```bash
mkdir -p data/acouslic-ai
mv acouslic-ai-train-set.zip data/acouslic-ai/
cd data/acouslic-ai
unzip acouslic-ai-train-set.zip
cd ../..
```

After extraction, the folder structure should look like:

```
data/
└── acouslic-ai/
    ├── circumferences/
    │   └── fetal_abdominal_circumferences_per_sweep.csv
    ├── images/
    │   └── stacked_fetal_ultrasound/
    │       └── *.mha
    └── masks/
        └── stacked_fetal_abdomen/
            └── *.mha
```

### Option B: Preprocessed Dataset
Alternatively, download the preprocessed data from:

  (update soon)

Place the zip file in the project root directory, then run:

```bash
mkdir -p data/acouslic-ai
mv acouslic-ai-train-set_preprocessed.zip data/acouslic-ai/
cd data/acouslic-ai
unzip acouslic-ai-train-set_preprocessed.zip
cd ../..
```

After extraction, the folder structure should look like:

```
data/
└── acouslic-ai/
    └── workshop/
        ├── train/
        │   └── *.npz
        ├── val/
        │   └── *.npz
        └── test/
            └── *.npz
```

---

## 2. Download Weights and Config

Download FetalCLIP model weights:

- [FetalCLIP weights](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/fadillah_maani_mbzuai_ac_ae/EspGREsyuOtEpxt36RoEUBoB6jtlsvPeoiDTBC1qX8WdZQ?e=uAbuyv)

Place the weight under the project root directory.

---

## 3. Preprocessing

**Skip this step if using preprocessed data.**

To preprocess raw data (train/val/test split and augmentation), run:

```bash
python preprocess.py
```

After preprocessing, the data folder structure will be:

```
data/
└── acouslic-ai/
    ├── circumferences/
    │   └── fetal_abdominal_circumferences_per_sweep.csv
    ├── images/
    │   └── stacked_fetal_ultrasound/
    │       └── *.mha
    ├── masks/
    │   └── stacked_fetal_abdomen/
    │       └── *.mha
    └── workshop/
        ├── train/
        │   └── *.npz
        ├── val/
        │   └── *.npz
        └── test/
            └── *.npz
```


Each `.npz` file contains:

```python
{
    "image": numpy.ndarray,
    "mask": numpy.ndarray
}
```

---

## 4. Reproduce Experiments

### Classification

```bash
python main.py --config config/classification.yml
```

Modify the `model_name` field in the YAML config file to experiment with different models.

### Segmentation

```bash
python main.py --config config/segmentation.yml
```

---

## Results
| Architecture    | Models                  | Accuracy    | F1 Score    | Precision    | Recall      | # Trainable<br>Parameters |
|-----------------|-------------------------|-------------|-------------|--------------|-------------|-------|
| **CNN**         | DenseNet                | 0.9516      | 0.7024      | 0.7805       | 0.6420      | 7.0 M |
|                 | EfficientNet            | 0.9537      | 0.7253      | 0.7725       | 0.6855      | 4.0 M |
|                 | VGG                     | 0.9510      | 0.7084      | 0.7580       | 0.6671      | 134 M |
| **Transformer** | Swin                    | 0.9565      | 0.7429      | **0.7864**   | 0.7113      | 1.7 M |
|                 | DEIT                    | 0.9554      | 0.7466      | 0.7619       | 0.7363      | 2.4 M |
|                 | ViT<sub>400M</sub>      | 0.9560      | 0.7506      | 0.7657       | **0.7417**  | 2.4 M |
|                 | FetalCLIP<sub>CLS</sub> | **0.9575**  | **0.7570**  | 0.7782       | 0.7397      | 2.4 M |

Model performance on fetal ultrasound image quality assessment (IQA). Metrics reported as mean over five runs. Best scores are **bolded**.

## Related Articles

```bibtex
@misc{he2025advancingfetalultrasoundimage,
      title={Advancing Fetal Ultrasound Image Quality Assessment in Low-Resource Settings}, 
      author={Dongli He and Hu Wang and Mohammad Yaqub},
      year={2025},
      eprint={2507.22802},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.22802}, 
}
```