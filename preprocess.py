from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from PIL import Image
from tqdm import tqdm, trange

SAVE_IMAGE_SIZE = 224


def train_val_test_split(
    image_dir: Path,
    mask_dir: Path,
    ac_csv_path: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    seed: int = 9872398152123123767126,
    visualize: bool=False,
):
    df = pd.read_csv(ac_csv_path)
    names = df["uuid"]

    rng = np.random.default_rng(seed)
    names = rng.permutation(names)

    # Split 70% train, 10& validation, and 20% test
    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
    num_uuids = len(names)
    train_val_idx = int(train_ratio * num_uuids)
    val_test_idx = int((1 - test_ratio) * num_uuids)
    train_names = names[:train_val_idx].tolist()
    val_names = names[train_val_idx:val_test_idx].tolist()
    test_names = names[val_test_idx:].tolist()

    train_aug = A.Compose([
        A.ColorJitter(0.2, 0.2, 0, 0, p=0.5),
        A.CLAHE(p=0.5),
        A.Affine(
            translate_percent=(-0.2, 0.2),
            scale=1.0,
            rotate=(-20, 20),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        ),
    ])

    general_aug = A.Compose(
        [
            A.Resize(
                SAVE_IMAGE_SIZE,
                SAVE_IMAGE_SIZE,
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=0,
                p=1.0,
            ),
        ]
    )

    num_stacks = 6
    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = row["uuid"]

        if name in train_names:
            target_dir = train_dir
            is_train = True
        elif name in val_names:
            target_dir = val_dir
            is_train = False
        elif name in test_names:
            target_dir = test_dir
            is_train = False

        image_path = image_dir / f"{name}.mha"
        mask_path = mask_dir / f"{name}.mha"

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        assert image_array.shape == mask_array.shape

        total_slices = mask_array.shape[0]
        slices_per_sweep = total_slices // num_stacks

        for stack in trange(num_stacks, leave=False, desc=f"Processing {name}"):
            col_name = f"sweep_{stack + 1}_ac_mm"
            if pd.isna(row[col_name]):
                continue  # Skip if AC is NaN

            # Determine slice range
            start = stack * slices_per_sweep
            end = (stack + 1) * slices_per_sweep

            for slice_idx in tqdm(range(start, end), leave=False):
                image = image_array[slice_idx]
                mask = mask_array[slice_idx]

                image = pad_to_square(image)
                mask = pad_to_square(mask)

                if is_train:
                    num_aug = 6
                else:
                    num_aug = 1
                for aug_idx in range(num_aug):
                    image_aug = image.copy()
                    mask_aug = mask.copy()
                    image_mask_aug = {"image": image_aug, "mask": mask_aug}

                    suffix = ""
                    if is_train and aug_idx > 0:
                        suffix = f"_{aug_idx - 1}"
                        image_mask_aug = train_aug(
                            image=image_mask_aug["image"], mask=image_mask_aug["mask"]
                        )

                    image_mask_aug = general_aug(
                        image=image_mask_aug["image"], mask=image_mask_aug["mask"]
                    )

                    # save image to png for visualization
                    filestem = f"{name}_{slice_idx}{suffix}"
                    if visualize:
                        img_pil = Image.fromarray(image_mask_aug["image"].astype(np.uint8))
                        save_path = (
                            target_dir / "image" / f"{filestem}.png"
                        )
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        img_pil.save(save_path)

                    save_path = target_dir / "data" / f"{filestem}.npz"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        save_path,
                        image=image_mask_aug["image"],
                        mask=image_mask_aug["mask"],
                    )


def pad_to_square(arr):
    h, w = arr.shape
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_arr = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    return padded_arr


if __name__ == "__main__":
    with open("config_cls.yml", "r") as file:
        config = yaml.safe_load(file)

    data_dir = Path(config["data_dir"])
    image_dir = data_dir / "images" / "stacked_fetal_ultrasound"
    mask_dir = data_dir / "masks" / "stacked_fetal_abdomen"
    ac_csv_path = (
        data_dir / "circumferences" / "fetal_abdominal_circumferences_per_sweep.csv"
    )
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_val_test_split(
        image_dir=image_dir,
        mask_dir=mask_dir,
        ac_csv_path=ac_csv_path,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        visualize=True,
    )
