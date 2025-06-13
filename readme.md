# Finetune FetalCLIP for Operable Ultrasound Frames Classification

## 0. Install enviroment
Python >= 3.10
Otherwise type hint will fail

## 1. Download Data

You can download the dataset from the this page:

https://zenodo.org/records/11005384

Or you can download the preprocessed data from

[this link](000)

once downloaded, put the file under the current directory, then perform

The data structure should be

Data
acouslic-ai
    circumferences
        fetal_abdominal_circumferences_per_sweep.csv
    images
        stacked_fetal_ultrasound
            *.mha
    masks
        stacked_fetal_abdomen
            *.mha
    train
        *.npz
    val
        *.npz
    test
        *.npz

npz contains
{"image": np.ndarray, "mask",: np.ndarray}

mkdir -p data/acouslic-ai
mv acouslic-ai-train-set.zip data/acouslic-ai/acouslic-ai-train-set.zip
cd data/acouslic-ai
unzip acouslic-ai-train-set.zip
cd ../..

Download data weights and FetalCLIP config file
[FetalCLIP weights](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/fadillah_maani_mbzuai_ac_ae/EspGREsyuOtEpxt36RoEUBoB6jtlsvPeoiDTBC1qX8WdZQ?e=uAbuyv)

[FetalCLIP config](https://raw.githubusercontent.com/BioMedIA-MBZUAI/FetalCLIP/refs/heads/main/FetalCLIP_config.json)

## 2. Preprocessing
In this step we performa train/val/test split and data augmenattion
If you download the raw data from Zenodo, you need to perform preprpcessing as the dataset has volumes contain no masks.

python preprocess.py
