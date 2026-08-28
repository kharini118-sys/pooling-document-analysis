import cv2
import os
import matplotlib.pyplot as plt


# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)


# Load the preprocessed document image
image = cv2.imread(
    "results/preprocessed.png",
    cv2.IMREAD_GRAYSCALE
)


if image is None:
    print("Error: Unable to load preprocessed image.")
    exit()


# Apply Canny Edge Detection
edges = cv2.Canny(
    image,
    50,
    150
)


# Save edge image
cv2.imwrite(
    "results/edges.png",
    edges
)


# Calculate number of detected edge pixels
edge_pixels = cv2.countNonZero(edges)

total_pixels = edges.shape[0] * edges.shape[1]

edge_percentage = (
    edge_pixels / total_pixels
) * 100


# Display original preprocessed image and edges
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Preprocessed Document")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Extracted Text Edges")
plt.axis("off")


plt.tight_layout()


# Save comparison
plt.savefig(
    "results/edge_comparison.png",
    dpi=200
)

plt.show()


print("Edge extraction completed successfully.")
print("Image shape:", edges.shape)
print("Total pixels:", total_pixels)
print("Detected edge pixels:", edge_pixels)
print(
    "Edge pixel percentage:",
    round(edge_percentage, 2),
    "%"
)