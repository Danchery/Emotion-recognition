
import os
from collections import Counter
from PIL import Image

def check_image_sizes(data_dir, folder, classes):
    sizes = []
    for cls in classes:
        class_path = os.path.join(data_dir, folder, cls)
        image_files = os.listdir(class_path)
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            sizes.append(img.size)

    unique_sizes = Counter(sizes)
    print(unique_sizes)
