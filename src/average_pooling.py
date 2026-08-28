import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)


def average_pooling(image, pool_size=2, stride=2):
    """
    Apply Average Pooling to a grayscale image.
    """

    height, width = image.shape

    output_height = (height - pool_size) // stride + 1
    output_width = (width - pool_size) // stride + 1

    pooled = np.zeros(
        (output_height, output_width),
        dtype=np.uint8
    )

    for i in range(output_height):

        for j in range(output_width):

            start_i = i * stride
            start_j = j * stride

            region = image[
                start_i:start_i + pool_size,
                start_j:start_j + pool_size
            ]

            pooled[i, j] = np.mean(region)

    return pooled


# Load extracted edge image
edges = cv2.imread(
    "results/edges.png",
    cv2.IMREAD_GRAYSCALE
)


if edges is None:
    print("Error: Edge image not found.")
    exit()


# Apply 2x2 Average Pooling
average_pooled = average_pooling(
    edges,
    pool_size=2,
    stride=2
)


# Save output
cv2.imwrite(
    "results/average_pooling.png",
    average_pooled
)


# Spatial resolution analysis
original_height, original_width = edges.shape

pooled_height, pooled_width = average_pooled.shape

original_pixels = original_height * original_width

pooled_pixels = pooled_height * pooled_width

reduction_percentage = (
    (1 - pooled_pixels / original_pixels) * 100
)


# Display comparison
plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.imshow(edges, cmap="gray")
plt.title(
    f"Original Edge Map\n{edges.shape}"
)
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(average_pooled, cmap="gray")
plt.title(
    f"2x2 Average Pooling\n{average_pooled.shape}"
)
plt.axis("off")


plt.tight_layout()


plt.savefig(
    "results/average_pooling_comparison.png",
    dpi=200
)

plt.show()


# Print analysis
print("\nAVERAGE POOLING ANALYSIS")
print("-" * 35)

print(
    "Original resolution:",
    original_height,
    "x",
    original_width
)

print(
    "Pooled resolution:",
    pooled_height,
    "x",
    pooled_width
)

print(
    "Original spatial positions:",
    original_pixels
)

print(
    "Pooled spatial positions:",
    pooled_pixels
)

print(
    "Spatial reduction:",
    round(reduction_percentage, 2),
    "%"
)