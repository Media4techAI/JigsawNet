import cv2
import sys

def print_image_sizes(img_path1, img_path2):
    # Load images
    img1 = cv2.imread(img_path1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED)

    if img1 is None:
        print(f"❌ Could not read image: {img_path1}")
        return
    if img2 is None:
        print(f"❌ Could not read image: {img_path2}")
        return

    # Shapes are (height, width, channels) or (height, width) if grayscale
    h1, w1 = img1.shape[:2]
    c1 = img1.shape[2] if len(img1.shape) == 3 else 1

    h2, w2 = img2.shape[:2]
    c2 = img2.shape[2] if len(img2.shape) == 3 else 1
    
    print(f"{img_path1}: {h1} x {w1} x {c1}")
    print(f"{img_path2}: {h2} x {w2} x {c2}")

if __name__ == "__main__":
    path_1 = "C:\\Users\\user\\Git\\JigsawNet\\Examples\\MIT_ex\\fragment_0001.png"
    path_2 = "C:\\Users\\user\\Git\\JigsawNet\\Examples\\MIT_ex\\fragment_0002.png"

    print_image_sizes(path_1, path_2)