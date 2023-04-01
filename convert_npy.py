import numpy as np
from PIL import Image
import os
import csv

# Path to the directory containing the image files
path_to_dir = "./datasets"
# Path to the CSV file containing the labels
path_to_labels = "./csv_files/label.csv"

# List all the image files in the directory
image_files = [os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir) if file.endswith(('.jpg', '.png'))]

# Initialize arrays to hold the images and labels
images = np.zeros((len(image_files), 64, 64))
labels = np.zeros(len(image_files))

# Load the labels from the CSV file
with open(path_to_labels, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for idx, row in enumerate(csvreader):
        # Set the label for the corresponding image file
        labels[idx] = int(row[0])

# Loop through the image files
for idx, image_file in enumerate(image_files):
    # Load the image
    img = Image.open(image_file)
    # Resize the image to 28x28
    img = img.resize((64, 64))
    # Convert the image to grayscale
    img = img.convert('L')
    # Convert the image to a NumPy array
    img = np.asarray(img)
    # Rescale the pixel values between 0 and 1
    img = img / 255.0
    # Save the image to the corresponding array
    images[idx] = img

# Save the image and label arrays as separate .npy files
np.save("images.npy", images)
np.save("labels.npy", labels)
