import cv2
import os
import matplotlib.pyplot as plt


# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)


# Load the document image
image = cv2.imread("dataset/sample_document.png")

if image is None:
    print("Error: Unable to load the document image.")
    exit()


# Convert image to grayscale
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Apply Gaussian Blur for basic noise reduction
blurred_image = cv2.GaussianBlur(
    gray_image,
    (3, 3),
    0
)


# Save preprocessing outputs
cv2.imwrite(
    "results/original.png",
    image
)

cv2.imwrite(
    "results/grayscale.png",
    gray_image
)

cv2.imwrite(
    "results/preprocessed.png",
    blurred_image
)


# Display comparison
plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Document")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(blurred_image, cmap="gray")
plt.title("Preprocessed Image")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/preprocessing_comparison.png",
    dpi=200
)

plt.show()


print("Preprocessing completed successfully.")
print("Original image shape:", image.shape)
print("Grayscale image shape:", gray_image.shape)
print("Preprocessed image shape:", blurred_image.shape)