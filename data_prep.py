import random
from pathlib import Path
from PIL import Image

IMG_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def resize_image(src_path, dst_path, size=IMG_SIZE):
    img = Image.open(src_path).convert("RGB")
    img = img.resize(size)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)


def split_files(files, ratios=SPLIT_RATIOS, seed=42):
    files = list(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def prepare_dataset(raw_dir="data/raw", out_dir="data/processed", classes=("cat", "dog")):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    for cls in classes:
        cls_files = sorted((raw_dir / cls).glob("*.*"))
        splits = split_files(cls_files)
        for split_name, files in splits.items():
            for f in files:
                dst = out_dir / split_name / cls / f.name
                resize_image(f, dst)
        train_n = len(splits["train"])
        val_n = len(splits["val"])
        test_n = len(splits["test"])
        print(f"[{cls}] train={train_n} val={val_n} test={test_n}")


if __name__ == "__main__":
    prepare_dataset()