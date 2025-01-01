from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image and convert it to a numpy array
image = np.array(Image.open('test2.png'))
w, h, d = image.shape

# Reshape the pixels into a 2D array (number of pixels x number of color channels)
pixels = np.reshape(image, (w * h, d)).astype(float)

# Create and fit the KMeans model
n_colors = 10
model = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)

# Extract the color palette (cluster centers)
palette = np.uint8(model.cluster_centers_)

# Create a figure to display both the original image and the palette
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # One row, two columns

# Display the original image
axes[0].imshow(image)
axes[0].axis('off')  # Remove axes for the image
axes[0].set_title("Original Image")

# Display the color palette
axes[1].imshow([palette])
axes[1].axis('off')  # Remove axes for the palette
axes[1].set_title("Color Palette")

plt.show()