import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


os.makedirs("results", exist_ok=True)


# -----------------------------------------
# LOAD FEATURE MAPS
# -----------------------------------------

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

    print("Error: Required images not found.")
    exit()


# -----------------------------------------
# CALCULATE METRICS
# -----------------------------------------

def calculate_metrics(name, image):

    height, width = image.shape

    total_pixels = height * width

    active_pixels = cv2.countNonZero(image)

    active_percentage = (
        active_pixels / total_pixels
    ) * 100

    mean_intensity = np.mean(image)

    return {
        "Method": name,
        "Resolution": f"{height} x {width}",
        "Total Positions": total_pixels,
        "Active Features": active_pixels,
        "Active Feature %": round(
            active_percentage,
            2
        ),
        "Mean Intensity": round(
            mean_intensity,
            2
        )
    }


# -----------------------------------------
# CREATE DATA
# -----------------------------------------

data = [

    calculate_metrics(
        "Original Edge Map",
        edges
    ),

    calculate_metrics(
        "2x2 Max Pooling",
        max_pooled
    ),

    calculate_metrics(
        "2x2 Average Pooling",
        average_pooled
    )

]


# -----------------------------------------
# CREATE DATAFRAME
# -----------------------------------------

df = pd.DataFrame(data)


print("\nFINAL QUANTITATIVE ANALYSIS\n")

print(df.to_string(index=False))


# -----------------------------------------
# SAVE CSV
# -----------------------------------------

df.to_csv(
    "results/final_metrics.csv",
    index=False
)


# -----------------------------------------
# BAR GRAPH: TOTAL POSITIONS
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Method"],
    df["Total Positions"]
)

plt.title(
    "Spatial Positions Before and After Pooling"
)

plt.ylabel(
    "Number of Spatial Positions"
)

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "results/spatial_resolution_graph.png",
    dpi=250
)

plt.show()


# -----------------------------------------
# BAR GRAPH: ACTIVE FEATURE %
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Method"],
    df["Active Feature %"]
)

plt.title(
    "Active Feature Percentage Comparison"
)

plt.ylabel(
    "Active Feature Percentage"
)

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "results/feature_retention_graph.png",
    dpi=250
)

plt.show()


print(
    "\nFinal analysis completed successfully."
)

print(
    "Metrics saved as results/final_metrics.csv"
)