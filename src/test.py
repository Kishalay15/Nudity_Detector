# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Compute the correct path relative to this script (inside src/)
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Goes up from 'src/' to project root
# train_csv_path = os.path.join(BASE_DIR, "data", "train_labels.csv")

# # Load the updated train_labels.csv
# df = pd.read_csv(train_csv_path)

# # Check counts of each label
# class_counts = df['label'].value_counts()
# print("Class Distribution:")
# print(class_counts)

# # Visualize the class distribution
# class_counts.plot(kind='bar')
# plt.title("Class Distribution in Train Dataset")
# plt.xlabel("Class")
# plt.ylabel("Number of Images")
# plt.xticks(rotation=0)
# plt.show()


import os
from PIL import Image

image_dir = os.path.join(os.getcwd(), "data", "train")  # Update if needed
threshold = 89478485  # PIL default pixel limit (85MP)

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, filename)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                pixels = w * h
                if pixels > threshold:
                    print(f"{filename} is oversized: {w}x{h} = {pixels} pixels")
        except Exception as e:
            print(f"Error with {filename}: {e}")
