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

https://zenodo.org/records/12697994

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
    ├── masks/
    │   └── stacked_fetal_abdomen/
    │       └── *.mha
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

Download model weights and FetalCLIP config file:

- [FetalCLIP weights](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/fadillah_maani_mbzuai_ac_ae/EspGREsyuOtEpxt36RoEUBoB6jtlsvPeoiDTBC1qX8WdZQ?e=uAbuyv)
- [FetalCLIP config](https://raw.githubusercontent.com/BioMedIA-MBZUAI/FetalCLIP/refs/heads/main/FetalCLIP_config.json)

Place both files under the project root directory.

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

### Segmentation

```bash
python main.py --config config/segmentation.yml
```

Modify the `model_name` field in the YAML config file to experiment with different models.

---

## Results
| Architecture | Models           | Accuracy ↑          | F1 Score ↑          | Precision ↑          | Recall ↑            | # Trainable Parameters |
|--------------|------------------|---------------------|---------------------|----------------------|---------------------|------------------------|
| **CNN**      | DenseNet         | 0.9516 ± 0.002      | 0.7024 ± 0.028      | _0.7805_ ± 0.026     | 0.6420 ± 0.059      | 7.0 M                  |
|              | EfficientNet     | 0.9537 ± 0.004      | 0.7253 ± 0.030      | 0.7725 ± 0.025       | 0.6855 ± 0.053      | 4.0 M                  |
|              | VGG              | 0.9510 ± 0.002      | 0.7084 ± 0.021      | 0.7580 ± 0.023       | 0.6671 ± 0.048      | 134 M                  |
| **Transformer** | Swin           | 0.9565 ± 0.003      | 0.7429 ± 0.039      | **0.7864** ± 0.032   | 0.7113 ± 0.087      | 1.7 M                  |
|              | DEIT             | 0.9554 ± 0.001      | 0.7466 ± 0.014      | 0.7619 ± 0.035       | 0.7363 ± 0.059      | 2.4 M                  |
|              | ViT<sub>400M</sub>          | _0.9560_ ± 0.003    | _0.7506_ ± 0.019    | 0.7657 ± 0.042       | **0.7417** ± 0.067  | 2.4 M                  |
|              | FetalCLIP<sub>CLS</sub>   | **0.9575** ± 0.001  | **0.7570** ± 0.007  | 0.7782 ± 0.034       | _0.7397_ ± 0.041    | 2.4 M                  |

*Table: Model performance on fetal ultrasound image quality assessment (IQA). Metrics reported as mean ± standard deviation over five runs. Best scores are **bolded**, second-best are _underlined_.*

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