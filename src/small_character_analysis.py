import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("results", exist_ok=True)


# -----------------------------------------
# LOAD ORIGINAL EDGE IMAGE
# -----------------------------------------

edges = cv2.imread(
    "results/edges.png",
    cv2.IMREAD_GRAYSCALE
)

if edges is None:
    print("Error: Edge image not found.")
    exit()


# -----------------------------------------
# MAX POOLING FUNCTION
# -----------------------------------------

def max_pooling(image, pool_size=2, stride=2):

    height, width = image.shape

    output_height = (
        (height - pool_size) // stride
    ) + 1

    output_width = (
        (width - pool_size) // stride
    ) + 1

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

            pooled[i, j] = np.max(region)

    return pooled


# -----------------------------------------
# TEXT REGIONS
# Coordinates: y1, y2, x1, x2
# -----------------------------------------

regions = {

    "Large Text": (40, 130, 30, 900),

    "Medium Text": (130, 210, 30, 900),

    "Small Text": (210, 280, 30, 700),

    "Very Small Text": (270, 330, 30, 700)

}


# -----------------------------------------
# CREATE VISUAL COMPARISON
# -----------------------------------------

plt.figure(figsize=(14, 12))


plot_number = 1


for name, coordinates in regions.items():

    y1, y2, x1, x2 = coordinates


    # Crop region
    crop = edges[y1:y2, x1:x2]


    # Apply pooling
    pooled_crop = max_pooling(
        crop,
        pool_size=2,
        stride=2
    )


    # Resize pooled result for visual comparison
    pooled_resized = cv2.resize(
        pooled_crop,
        (crop.shape[1], crop.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


    # Original crop
    plt.subplot(4, 2, plot_number)

    plt.imshow(
        crop,
        cmap="gray"
    )

    plt.title(
        name + " - Original Edges"
    )

    plt.axis("off")

    plot_number += 1


    # Pooled crop
    plt.subplot(4, 2, plot_number)

    plt.imshow(
        pooled_resized,
        cmap="gray"
    )

    plt.title(
        name + " - After 2x2 Max Pooling"
    )

    plt.axis("off")

    plot_number += 1


plt.tight_layout()


plt.savefig(
    "results/small_character_analysis.png",
    dpi=250
)


plt.show()


# -----------------------------------------
# CALCULATE FEATURE RETENTION
# -----------------------------------------

print("\nSMALL CHARACTER PRESERVATION ANALYSIS")

print("=" * 45)


for name, coordinates in regions.items():

    y1, y2, x1, x2 = coordinates

    crop = edges[y1:y2, x1:x2]

    pooled_crop = max_pooling(crop)


    original_active = cv2.countNonZero(crop)

    pooled_active = cv2.countNonZero(
        pooled_crop
    )


    print("\n", name)

    print(
        "Original resolution:",
        crop.shape
    )

    print(
        "Pooled resolution:",
        pooled_crop.shape
    )

    print(
        "Original active edges:",
        original_active
    )

    print(
        "Pooled active edges:",
        pooled_active
    )

    print(
        "Observation: Spatial detail is reduced."
    )


print(
    "\nSmall character analysis completed successfully."
)
with open(
    "results/small_character_metrics.txt",
    "w"
) as file:

    file.write(
        "SMALL CHARACTER PRESERVATION ANALYSIS\n"
    )

    file.write(
        "=" * 45 + "\n\n"
    )

    for name, coordinates in regions.items():

        y1, y2, x1, x2 = coordinates

        crop = edges[y1:y2, x1:x2]

        pooled_crop = max_pooling(crop)

        original_active = cv2.countNonZero(crop)

        pooled_active = cv2.countNonZero(
            pooled_crop
        )

        file.write(
            name + "\n"
        )

        file.write(
            "Original resolution: "
            + str(crop.shape)
            + "\n"
        )

        file.write(
            "Pooled resolution: "
            + str(pooled_crop.shape)
            + "\n"
        )

        file.write(
            "Original active edges: "
            + str(original_active)
            + "\n"
        )

        file.write(
            "Pooled active edges: "
            + str(pooled_active)
            + "\n\n"
        )


print(
    "Metrics saved successfully."
)