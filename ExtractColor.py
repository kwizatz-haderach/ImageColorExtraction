from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image and convert it to a numpy array
image = np.array(Image.open('test.png'))
w, h, d = image.shape

# Reshape the pixels into a 2D array (number of pixels x number of color channels)
pixels = np.reshape(image, (w * h, d)).astype(float)

# Create and fit the KMeans model
n_colors = 10
model = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)

# Extract the color palette (cluster centers)
palette = np.uint8(model.cluster_centers_)

# Visualize the color palette
plt.imshow([palette])
plt.axis('off')  # Remove unnecessary axes
plt.show()