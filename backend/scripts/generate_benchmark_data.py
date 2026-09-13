import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_data"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

def create_synthetic_passport_image(filename: str, name: str, pass_num: str, photo_color: tuple = (100, 150, 200)) -> Path:
    """Creates a synthetic passport document image."""
    img = np.full((600, 850, 3), (240, 240, 240), dtype=np.uint8)
    
    # Draw header band
    cv2.rectangle(img, (0, 0), (850, 80), (30, 40, 100), -1)
    cv2.putText(img, "PASSPORT - REPUBLIC OF INDIA", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Draw photo box
    cv2.rectangle(img, (40, 120), (240, 380), photo_color, -1)
    cv2.putText(img, "PHOTO", (90, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw synthetic face circle inside photo box
    cv2.circle(img, (140, 220), 45, (220, 180, 140), -1) # face
    cv2.circle(img, (125, 210), 6, (40, 40, 40), -1)     # left eye
    cv2.circle(img, (155, 210), 6, (40, 40, 40), -1)     # right eye
    cv2.ellipse(img, (140, 240), (18, 10), 0, 0, 180, (40, 40, 40), 3) # smile
    
    # Draw text fields
    cv2.putText(img, f"Type: P   Code: IND   Passport No: {pass_num}", (270, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(img, f"Name / Name: {name}", (270, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(img, "Nationality: INDIAN", (270, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(img, "Date of Birth: 15/05/1995", (270, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(img, "Sex: M", (270, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(img, "Date of Expiry: 20/12/2032", (270, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    
    # Draw MRZ zone
    mrz_line1 = f"P<IND{name.replace(' ', '<').upper()}<<<<<<<<<<<<<<<<<<<<<<<<"[:44]
    mrz_line2 = f"{pass_num}0IND9505154M3212204<<<<<<<<<<<<<<02"[:44]
    
    cv2.rectangle(img, (20, 470), (830, 580), (220, 220, 220), -1)
    cv2.putText(img, mrz_line1, (30, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)
    cv2.putText(img, mrz_line2, (30, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)

    out_path = BENCHMARK_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path

def create_messaging_app_recompressed_image(input_path: Path, output_filename: str) -> Path:
    """Resizes input image to 1600x1200 with stripped EXIF (simulates WhatsApp)."""
    img = cv2.imread(str(input_path))
    resized = cv2.resize(img, (1600, 1200))
    out_path = BENCHMARK_DIR / output_filename
    cv2.imwrite(str(out_path), resized)
    return out_path

def create_photoshop_tagged_image(input_path: Path, output_filename: str) -> Path:
    """Saves image with Software: Adobe Photoshop EXIF metadata tag."""
    pil_img = Image.open(input_path)
    exif = pil_img.getexif()
    # Tag 305 is Software
    exif[305] = "Adobe Photoshop 2024 (Windows)"
    out_path = BENCHMARK_DIR / output_filename
    pil_img.save(out_path, exif=exif)
    return out_path

def create_live_face_image(filename: str, face_color: tuple = (220, 180, 140)) -> Path:
    """Creates a synthetic live capture face image."""
    img = np.full((320, 320, 3), (200, 200, 200), dtype=np.uint8)
    cv2.circle(img, (160, 160), 80, face_color, -1)     # face
    cv2.circle(img, (130, 140), 10, (40, 40, 40), -1)    # left eye
    cv2.circle(img, (190, 140), 10, (40, 40, 40), -1)    # right eye
    cv2.ellipse(img, (160, 190), (30, 15), 0, 0, 180, (40, 40, 40), 4) # smile
    out_path = BENCHMARK_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path

def main():
    print("Generating benchmark test dataset...")
    
    # 1. Genuine passport
    p1 = create_synthetic_passport_image("genuine_passport_01.jpg", "PARTH SONI", "Z1234567")
    
    # 2. Tampered passport (spliced block noise)
    p2_img = cv2.imread(str(p1))
    # Overlay high contrast noise box over photo area
    noise_box = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    p2_img[150:250, 50:150] = noise_box
    p2 = BENCHMARK_DIR / "tampered_passport_01.jpg"
    cv2.imwrite(str(p2), p2_img)
    
    # 3. WhatsApp re-compressed genuine passport
    p3 = create_messaging_app_recompressed_image(p1, "genuine_whatsapp_1600x1200.jpg")
    
    # 4. Photoshop tagged tampered image
    p4 = create_photoshop_tagged_image(p2, "tampered_photoshop.jpg")
    
    # 5. Live face capture images
    face_match = create_live_face_image("live_capture_match.jpg", (220, 180, 140))
    face_mismatch = create_live_face_image("live_capture_mismatch.jpg", (100, 80, 50))
    
    # Labels metadata dictionary
    labels = {
        "tampering_cases": [
            {
                "path": str(p1),
                "document_type": "Passport",
                "ground_truth": "genuine"
            },
            {
                "path": str(p2),
                "document_type": "Passport",
                "ground_truth": "tampered"
            },
            {
                "path": str(p3),
                "document_type": "Passport",
                "ground_truth": "genuine"
            },
            {
                "path": str(p4),
                "document_type": "Passport",
                "ground_truth": "tampered"
            }
        ],
        "ocr_cases": [
            {
                "path": str(p1),
                "document_type": "Passport",
                "expected_fields": {
                    "document_number": "Z1234567"
                }
            }
        ],
        "face_cases": [
            {
                "document_path": str(p1),
                "live_path": str(face_match),
                "document_type": "Passport",
                "ground_truth": "match"
            },
            {
                "document_path": str(p1),
                "live_path": str(face_mismatch),
                "document_type": "Passport",
                "ground_truth": "mismatch"
            }
        ]
    }
    
    labels_file = BENCHMARK_DIR / "labels.json"
    labels_file.write_text(json.dumps(labels, indent=2))
    print(f"Benchmark dataset successfully generated at {BENCHMARK_DIR}")

if __name__ == "__main__":
    main()
