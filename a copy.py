import os
import subprocess
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def pptx_to_images(pptx_path: str, output_dir: str, dpi: int = 200):
    """
    Converts every slide in a .pptx into a separate PNG using LibreOffice & Poppler.
    Requires:
      • LibreOffice (soffice) on your PATH
      • poppler-utils (for pdf2image.convert_from_path)
      • pip install python-pptx pdf2image
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert PPTX → PDF via LibreOffice
    subprocess.run([
        "soffice", "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        pptx_path
    ], check=True)

    # Locate the generated PDF
    base_name = os.path.splitext(os.path.basename(pptx_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Expected PDF at {pdf_path}")

    # Convert PDF pages to PNGs
    slides = convert_from_path(pdf_path, dpi=dpi)
    for idx, img in enumerate(slides, start=1):
        out_file = os.path.join(output_dir, f"slide_{idx}.png")
        img.save(out_file, "PNG")
        print(f"Written {out_file}")

    # Optional: remove intermediate PDF
    # os.remove(pdf_path)


def extract_text_from_images(output_dir: str, max_workers: int = 5):
    """
    Concurrently extracts text from each PNG in output_dir using Tesseract OCR,
    and prints a combined report.
    """
    # Gather all slide image filenames
    slides = sorted([f for f in os.listdir(output_dir) if f.lower().endswith('.png')])

    def process_slide(img_name: str):
        img_path = os.path.join(output_dir, img_name)
        text = pytesseract.image_to_string(Image.open(img_path))
        return img_name, text.strip()

    # Run concurrent extraction
    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(slides))) as executor:
        for img_name, text in executor.map(process_slide, slides):
            results.append((img_name, text))

    # Print final combined output
    for img_name, text in results:
        slide_num = img_name.replace('slide_', '').replace('.png', '')
        print(f"slide{slide_num}:{text}")
    print("END")


if __name__ == "__main__":
    # Adjust these paths as needed
    pptx_path = "/Users/priyamsekra/my_code/DATA 2/novus-vista/Bajaj-hack/bajab_finance_document_retirval_contest/Test Case HackRx.pptx"
    output_dir = "/Users/priyamsekra/my_code/DATA 2/novus-vista/Bajaj-hack/bajab_finance_document_retirval_contest/slide_images"

    # 1) Export slides as images
    pptx_to_images(pptx_path, output_dir)

    # 2) Concurrently extract text and print each slide's content
    extract_text_from_images(output_dir, max_workers=5)