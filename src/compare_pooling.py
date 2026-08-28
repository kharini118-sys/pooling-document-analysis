import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("results", exist_ok=True)


# Load images
edges = cv2.imread(
    "results/edges.png",
    cv2.IMREAD_GRAYSCALE
)

max_pooled = cv2.imread(
    "results/max_pooling.png",
    cv2.IMREAD_GRAYSCALE
)

average_pooled = cv2.imread(
    "results/average_pooling.png",
    cv2.IMREAD_GRAYSCALE
)


if edges is None or max_pooled is None or average_pooled is None:
    print("Error: One or more images could not be loaded.")
    exit()


# ------------------------------------------------
# FUNCTION TO CALCULATE FEATURE INFORMATION
# ------------------------------------------------

def calculate_metrics(image):

    total_pixels = image.shape[0] * image.shape[1]

    active_pixels = cv2.countNonZero(image)

    active_percentage = (
        active_pixels / total_pixels
    ) * 100

    mean_intensity = np.mean(image)

    return {
        "resolution": image.shape,
        "total_pixels": total_pixels,
        "active_pixels": active_pixels,
        "active_percentage": active_percentage,
        "mean_intensity": mean_intensity
    }


# Calculate metrics
original_metrics = calculate_metrics(edges)

max_metrics = calculate_metrics(max_pooled)

average_metrics = calculate_metrics(average_pooled)


# ------------------------------------------------
# PRINT ANALYSIS
# ------------------------------------------------

print("\nPOOLING COMPARISON ANALYSIS")
print("=" * 45)


print("\nORIGINAL EDGE MAP")

for key, value in original_metrics.items():

    if isinstance(value, float):
        print(key, ":", round(value, 2))

    else:
        print(key, ":", value)


print("\nMAX POOLED EDGE MAP")

for key, value in max_metrics.items():

    if isinstance(value, float):
        print(key, ":", round(value, 2))

    else:
        print(key, ":", value)


print("\nAVERAGE POOLED EDGE MAP")

for key, value in average_metrics.items():

    if isinstance(value, float):
        print(key, ":", round(value, 2))

    else:
        print(key, ":", value)


# ------------------------------------------------
# SPATIAL RESOLUTION REDUCTION
# ------------------------------------------------

original_pixels = original_metrics["total_pixels"]

max_reduction = (
    1 -
    max_metrics["total_pixels"] /
    original_pixels
) * 100


average_reduction = (
    1 -
    average_metrics["total_pixels"] /
    original_pixels
) * 100


print("\nSPATIAL RESOLUTION REDUCTION")

print(
    "Max Pooling:",
    round(max_reduction, 2),
    "%"
)

print(
    "Average Pooling:",
    round(average_reduction, 2),
    "%"
)


# ------------------------------------------------
# VISUAL COMPARISON
# ------------------------------------------------

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(edges, cmap="gray")

plt.title(
    f"Original Edge Map\n{edges.shape}"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(max_pooled, cmap="gray")

plt.title(
    f"2x2 Max Pooling\n{max_pooled.shape}"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(average_pooled, cmap="gray")

plt.title(
    f"2x2 Average Pooling\n{average_pooled.shape}"
)

plt.axis("off")


plt.tight_layout()


plt.savefig(
    "results/pooling_comparison.png",
    dpi=250
)

plt.show()


# ------------------------------------------------
# SAVE METRICS TO TEXT FILE
# ------------------------------------------------

with open(
    "results/pooling_analysis.txt",
    "w"
) as file:

    file.write(
        "POOLING COMPARISON ANALYSIS\n"
    )

    file.write(
        "=" * 40 + "\n\n"
    )


    file.write(
        "Original Edge Map\n"
    )

    file.write(
        str(original_metrics) + "\n\n"
    )


    file.write(
        "Max Pooling\n"
    )

    file.write(
        str(max_metrics) + "\n\n"
    )


    file.write(
        "Average Pooling\n"
    )

    file.write(
        str(average_metrics) + "\n\n"
    )


    file.write(
        "Spatial Resolution Reduction\n"
    )

    file.write(
        "Max Pooling: "
        + str(round(max_reduction, 2))
        + "%\n"
    )

    file.write(
        "Average Pooling: "
        + str(round(average_reduction, 2))
        + "%\n"
    )


print("\nAnalysis completed successfully.")

print(
    "Results saved in results/pooling_analysis.txt"
)