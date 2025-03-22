import os
import pandas as pd
import matplotlib.pyplot as plt

# Compute the correct path relative to this script (inside src/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Goes up from 'src/' to project root
train_csv_path = os.path.join(BASE_DIR, "data", "train_labels.csv")

# Load the updated train_labels.csv
df = pd.read_csv(train_csv_path)

# Check counts of each label
class_counts = df['label'].value_counts()
print("Class Distribution:")
print(class_counts)

# Visualize the class distribution
class_counts.plot(kind='bar')
plt.title("Class Distribution in Train Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.show()
