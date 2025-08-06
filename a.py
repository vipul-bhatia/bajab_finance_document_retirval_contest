import os
import subprocess
import base64
from pdf2image import convert_from_path
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


def encode_image(image_path: str) -> str:
    """
    Reads an image file and returns a Base64-encoded data URL.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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


def summarize_images(output_dir: str, max_workers: int = 25):
    """
    Concurrently encodes each PNG in output_dir to Base64 and requests a 2-line summary from OpenAI.
    """
    load_dotenv()  # loads OPENAI_API_KEY
    client = OpenAI()

    # Gather all slide image filenames
    slides = sorted([f for f in os.listdir(output_dir) if f.lower().endswith('.png')])

    def process_slide(img_name: str):
        img_path = os.path.join(output_dir, img_name)
        print(f"\nSummarizing {img_name}...")
        data_url = encode_image(img_path)
        response = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Please extract the text from the image as is. Do not summarize or paraphrase. If no text present, then summarize the image in 2 lines."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
        )
        return img_name, response.output_text

    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=min(max_workers, len(slides))) as executor:
        for img_name, summary in executor.map(process_slide, slides):
            print(f"Summary for {img_name}:\n{summary}")


if __name__ == "__main__":
    # Adjust these paths as needed
    pptx_path = "/Users/priyamsekra/my_code/DATA 2/novus-vista/Bajaj-hack/bajab_finance_document_retirval_contest/Test Case HackRx.pptx"
    output_dir = "/Users/priyamsekra/my_code/DATA 2/novus-vista/Bajaj-hack/bajab_finance_document_retirval_contest/slide_images"


    # 1) Export slides as images
    pptx_to_images(pptx_path, output_dir)

    # 2) Concurrently summarize each slide image
    summarize_images(output_dir, max_workers=25)




