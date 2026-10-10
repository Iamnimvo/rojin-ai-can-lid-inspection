
from pathlib import Path
from PIL import Image, ImageOps
import pillow_avif

ROOT = Path(r"D:\rojin_lid_ai")

INPUT_DIR = ROOT / "data" / "rojin_test"
OUTPUT_DIR = ROOT / "data" / "rojin_test_png"

SUPPORTED = {
    ".jpg", ".jpeg", ".png", ".webp", ".avif"
}

EXPECTED = {
    "good": 3,
    "defective": 4
}


def convert_image(source, destination):

    with Image.open(source) as image:

        image.load()

        image = ImageOps.exif_transpose(image)

        # Handle transparency with a white background
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:

            rgba = image.convert("RGBA")

            background = Image.new(
                "RGBA", rgba.size, (255, 255, 255, 255)
            )

            background.alpha_composite(rgba)

            image = background.convert("RGB")

        else:
            image = image.convert("RGB")

        if min(image.size) < 224:
            print(f"WARNING: Low resolution: {source.name}")

        image.save(destination, format="PNG")


def main():

    print("=" * 60)
    print("ROJIN EXTERNAL DATASET PREPARATION")
    print("=" * 60)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    for class_name, expected_count in EXPECTED.items():

        source_folder = INPUT_DIR / class_name
        target_folder = OUTPUT_DIR / class_name

        target_folder.mkdir(parents=True, exist_ok=True)

        if not source_folder.exists():
            raise FileNotFoundError(source_folder)

        files = sorted([
            p for p in source_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED
        ])

        print(f"\nClass: {class_name}")
        print(f"Found: {len(files)} images")

        if len(files) != expected_count:
            raise ValueError(
                f"Expected {expected_count} images "
                f"but found {len(files)} in {source_folder}"
            )

        # Avoid retaining output from a previous preparation
        for old_file in target_folder.glob("*.png"):
            old_file.unlink()

        for index, source in enumerate(files, start=1):

            filename = f"{class_name}_{index:02d}.png"

            destination = target_folder / filename

            convert_image(source, destination)

            # Verify that the converted file is readable
            with Image.open(destination) as check:
                check.verify()

            print(f"OK: {source.name} -> {filename}")

    print("\n" + "=" * 60)
    print("ALL 7 IMAGES CONVERTED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()