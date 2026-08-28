import cv2
import numpy as np
import os


# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)


# Document dimensions
width = 1000
height = 700

# Create white document
image = np.ones((height, width), dtype=np.uint8) * 255

font = cv2.FONT_HERSHEY_SIMPLEX


# Large characters
cv2.putText(
    image,
    "DOCUMENT SCANNING",
    (50, 100),
    font,
    1.5,
    0,
    3,
    cv2.LINE_AA
)


# Medium characters
cv2.putText(
    image,
    "Deep Learning Pooling Analysis",
    (50, 180),
    font,
    0.9,
    0,
    2,
    cv2.LINE_AA
)


# Small characters
cv2.putText(
    image,
    "Small characters: ABC123 xyz789",
    (50, 250),
    font,
    0.5,
    0,
    1,
    cv2.LINE_AA
)


# Very small characters
cv2.putText(
    image,
    "Fine text 12 34 56 A B C",
    (50, 300),
    font,
    0.3,
    0,
    1,
    cv2.LINE_AA
)


# Document-like horizontal lines
for y in range(360, 600, 45):
    cv2.line(
        image,
        (50, y),
        (900, y),
        0,
        1
    )


# Save image
output_path = "dataset/sample_document.png"

cv2.imwrite(output_path, image)

print("Sample document created successfully.")
print("Saved to:", output_path)