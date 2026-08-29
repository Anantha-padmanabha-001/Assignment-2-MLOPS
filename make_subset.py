import os, shutil, random

os.makedirs("data/raw/cat", exist_ok=True)
os.makedirs("data/raw/dog", exist_ok=True)

def copy_subset(src, dst, n=300):
    files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    copied = 0
    for f in files[:n]:
        try:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            copied += 1
        except Exception:
            pass
    return copied

print("cat:", copy_subset("PetImages/Cat", "data/raw/cat", 300))
print("dog:", copy_subset("PetImages/Dog", "data/raw/dog", 300))