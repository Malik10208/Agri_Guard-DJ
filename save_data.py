import os, shutil, random

# paste the path from the command above here
SOURCE = r"C:\Users\Abdul Malik\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"

DEST_TRAIN_H  = r"C:\Main folder\Agri gaurd\data\train\healthy"
DEST_TRAIN_D  = r"C:\Main folder\Agri gaurd\data\train\diseased"
DEST_VAL_H    = r"C:\Main folder\Agri gaurd\data\val\healthy"
DEST_VAL_D    = r"C:\Main folder\Agri gaurd\data\val\diseased"

os.makedirs(DEST_VAL_H, exist_ok=True)
os.makedirs(DEST_VAL_D, exist_ok=True)

# Find healthy and diseased folders inside PlantVillage
for folder in os.listdir(SOURCE):
    full = os.path.join(SOURCE, folder)
    if not os.path.isdir(full):
        continue

    images = [f for f in os.listdir(full) if f.endswith('.jpg')]
    random.shuffle(images)
    pick = images[:150]  # take 150 images per class
    train_imgs = pick[:120]
    val_imgs   = pick[120:]

    if 'healthy' in folder.lower():
        dest_t, dest_v = DEST_TRAIN_H, DEST_VAL_H
    else:
        dest_t, dest_v = DEST_TRAIN_D, DEST_VAL_D

    for f in train_imgs:
        shutil.copy(os.path.join(full, f), dest_t)
    for f in val_imgs:
        shutil.copy(os.path.join(full, f), dest_v)

print("Done! Images copied.")