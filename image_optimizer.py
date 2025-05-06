import os
import cv2
import numpy as np

def generate_leaf_contour_image(input_path, output_path, target_size=(224, 224)):
    # 1. Kép beolvasása
    img = cv2.imread(input_path)
    if img is None:
        print(f"Hiba: {input_path} nem tölthető be.")
        return False

    # 2. Szürkeárnyalatos + elmosás
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Küszöbölés + kontúr keresés
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Kontúrok rajzolása fekete háttérre
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness=2)

    # 5. Átméretezés
    resized = cv2.resize(contour_img, target_size)

    # 6. Mentés
    cv2.imwrite(output_path, resized)
    return True


def generate_contour_dataset(input_root, output_root, target_size=(224, 224)):
    count_by_class = {}
    total = 0

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)

                # Kategória neve mappa alapján
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, file)

                success = generate_leaf_contour_image(input_path, output_path, target_size)

                if success:
                    total += 1
                    count_by_class[relative_path] = count_by_class.get(relative_path, 0) + 1

    # Eredmények kiírása
    print(f"\n✅ Összesen {total} képet dolgoztunk fel.")
    for cls, count in sorted(count_by_class.items()):
        print(f"  └── {cls}: {count} kép")


# 🔧 Itt állítsd be az elérési útvonalakat
if __name__ == "__main__":
    input_root = "D:/Documents/Suli/PE/fejlett-kepfeldolgozas/kepek/levelek/"      # Pl. 'data/raw' vagy bármi más
    output_root = "D:/Documents/Suli/PE/fejlett-kepfeldolgozas/kepek/f_levelek/"      # A generált képek ide kerülnek

    generate_contour_dataset(input_root, output_root)
