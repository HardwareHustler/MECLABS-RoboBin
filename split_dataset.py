import os
import shutil
import random

SOURCE_DIR = "dataset-resized"
TARGET_DIR = "dataset"
SPLIT_RATIO = 0.8

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:
    src_cls_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src_cls_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(TARGET_DIR, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "val", cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(src_cls_path, img),
            os.path.join(TARGET_DIR, "train", cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(src_cls_path, img),
            os.path.join(TARGET_DIR, "val", cls, img)
        )

print("✅ Dataset split completed successfully")
