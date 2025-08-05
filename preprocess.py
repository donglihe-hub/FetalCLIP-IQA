from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
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
    num_train_aug: int = 2,
    visualize: bool = False,
):
    df = pd.read_csv(ac_csv_path)
    uuids = df["uuid"]

    rng = np.random.default_rng(seed)
    uuids = rng.permutation(uuids)

    # Split 70% train, 10& validation, and 20% test
    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
    num_uuids = len(uuids)
    train_val_idx = int(train_ratio * num_uuids)
    val_test_idx = int((1 - test_ratio) * num_uuids)
    train_uuids = uuids[:train_val_idx].tolist()
    val_uuids = uuids[train_val_idx:val_test_idx].tolist()
    test_uuids = uuids[val_test_idx:].tolist()

    train_aug = A.Compose(
        [
            A.ColorJitter(),
            A.CLAHE(),
            A.Affine(
                scale=1.0,
                translate_percent=(-0.2, 0.2),
                rotate=(-20, 20),
                shear=0.0,
                p=1.0,
            ),
        ]
    )

    general_aug = A.Compose(
        [
            A.Resize(SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
        ]
    )

    num_stacks = 6
    for _, row in tqdm(df.iterrows(), total=len(df)):
        uuid = row["uuid"]

        if uuid in train_uuids:
            target_dir = train_dir
            is_train = True
        elif uuid in val_uuids:
            target_dir = val_dir
            is_train = False
        elif uuid in test_uuids:
            target_dir = test_dir
            is_train = False

        image_path = image_dir / f"{uuid}.mha"
        mask_path = mask_dir / f"{uuid}.mha"

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        assert image_array.shape == mask_array.shape

        total_slices = mask_array.shape[0]  # 720
        slices_per_sweep = total_slices // num_stacks  # 120

        for stack in trange(num_stacks, leave=False, desc=f"Processing {uuid}"):
            col_name = f"sweep_{stack + 1}_ac_mm"
            if pd.isna(row[col_name]):
                continue  # skip if AC is NaN

            # Determine slice range
            start = stack * slices_per_sweep
            end = (stack + 1) * slices_per_sweep

            for slice_idx in trange(start, end, leave=False):
                image = image_array[slice_idx]
                mask = mask_array[slice_idx]

                image = pad_to_square(image)
                mask = pad_to_square(mask)

                image_mask = {"image": image, "mask": mask}
                image_mask = general_aug(
                    image=image_mask["image"], mask=image_mask["mask"]
                )
                output_path = target_dir / f"{uuid}_{slice_idx}.npz"
                save_image_and_mask(output_path, image_mask, visualize)

                # Apply training augmentations
                if is_train:
                    for aug_idx in range(num_train_aug):
                        image_mask_aug = {"image": image, "mask": mask}

                        image_mask_aug = train_aug(
                            image=image_mask_aug["image"], mask=image_mask_aug["mask"]
                        )
                        image_mask_aug = general_aug(
                            image=image_mask_aug["image"], mask=image_mask_aug["mask"]
                        )
                        output_path = (
                            target_dir / f"{uuid}_{slice_idx}_{aug_idx + 1}.npz"
                        )
                        save_image_and_mask(output_path, image_mask_aug, visualize)


def save_image_and_mask(
    output_path: Path, image_mask_aug: dict[str : np.ndarray], visualize: bool
):
    # save image to png for visualization
    if visualize:
        img_pil = Image.fromarray(image_mask_aug["image"].astype(np.uint8))
        image_save_path = (
            output_path.parent.parent / "image" / f"{output_path.stem}.png"
        )
        image_save_path.parent.mkdir(parents=True, exist_ok=True)
        img_pil.save(image_save_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        image=image_mask_aug["image"],
        mask=image_mask_aug["mask"],
    )


def pad_to_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    max_size = max(h, w)
    top = (max_size - h) // 2
    left = (max_size - w) // 2

    square_arr = np.zeros((max_size, max_size), dtype=arr.dtype)
    square_arr[top : top + h, left : left + w] = arr

    return square_arr


def generate_meta_info(output_dir: Path):
    """
    Generate a CSV file with the label information for the specified split.
    The CSV will contain the split, filename and label (1 for presence of mask, 0 for absence).
    """
    meta_info_list = []
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        for npz_file in tqdm(
            sorted(split_dir.glob("*.npz")),
            desc=f"Collecting meta info for {split} data",
        ):
            data = np.load(npz_file)
            label = (data["mask"].max() > 0).astype(int)
            meta_info_list.append(
                {"split": split, "filename": npz_file.stem, "label": label}
            )

    df = pd.DataFrame(meta_info_list)
    df.to_csv(output_dir / "meta_info.csv", index=False)


if __name__ == "__main__":
    # input directories
    data_dir = Path("data") / "acouslic-ai"
    image_dir = data_dir / "images" / "stacked_fetal_ultrasound"
    mask_dir = data_dir / "masks" / "stacked_fetal_abdomen"
    ac_csv_path = (
        data_dir / "circumferences" / "fetal_abdominal_circumferences_per_sweep.csv"
    )

    # output directories
    output_dir = data_dir / "workshop"
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    train_val_test_split(
        image_dir=image_dir,
        mask_dir=mask_dir,
        ac_csv_path=ac_csv_path,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        num_train_aug=2,
        visualize=False,  # Set to True if you want to save images for visualization
    )
    # generate_meta_info(output_dir)
