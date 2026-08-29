import sys
from io import BytesIO
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from data_prep import split_files

def test_split_files_ratios_roughly_correct():
    files = [f"img_{i}.jpg" for i in range(100)]
    result = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    assert len(result["train"]) == 80
    assert len(result["val"]) == 10
    assert len(result["test"]) == 10

def test_split_files_no_overlap_and_covers_all_files():
    files = [f"img_{i}.jpg" for i in range(50)]
    result = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    all_split = result["train"] + result["val"] + result["test"]
    assert sorted(all_split) == sorted(files)
    assert len(set(all_split)) == len(files)

def test_split_files_deterministic_with_same_seed():
    files = [f"img_{i}.jpg" for i in range(30)]
    r1 = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    r2 = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    assert r1 == r2

def _fake_jpeg_bytes(size=(300, 300), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_model_predicts_binary_class():
    from main import SimpleCNN, TRANSFORM, CLASS_NAMES
    model = SimpleCNN(num_classes=2)
    model.eval()
    img = Image.open(BytesIO(_fake_jpeg_bytes())).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = int(logits.argmax(dim=1))
    assert CLASS_NAMES[pred] in ("cat", "dog")

def test_model_returns_valid_probabilities():
    from main import SimpleCNN, TRANSFORM
    model = SimpleCNN(num_classes=2)
    model.eval()
    img = Image.open(BytesIO(_fake_jpeg_bytes())).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 0.01