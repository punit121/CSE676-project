import os
import gdown
import zipfile
import tarfile
from pathlib import Path

# Setup directories
BASE_DIR = "/content"
DATA_DIR = os.path.join(BASE_DIR, "data")
FAKEDDIT_DIR = os.path.join(DATA_DIR, "Fakeddit")
IMAGES_DIR = os.path.join(FAKEDDIT_DIR, "images")

os.makedirs(FAKEDDIT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

print("📁 Directories created:")
print(f"  - Fakeddit directory: {FAKEDDIT_DIR}")
print(f"  - Images directory: {IMAGES_DIR}")



# Extract file IDs from the Google Drive URLs
TEXT_TEST_FILE_ID = "1ExPiA_v2Dq_rG6afY4rvgUV-Jh9ByiTI"
IMAGE_TEST_FILE_ID = "1YtvM2Muf4hT0SCALI7FaILaGPJdW7lW1"


def install_gdown():
    """Install gdown for Google Drive downloads"""
    os.system("pip install -q gdown")


def download_text_test_set():
    """Download the private text test set"""

    url = f"https://drive.google.com/uc?id={TEXT_TEST_FILE_ID}"
    output_path = os.path.join(FAKEDDIT_DIR, "text_test.zip")



    try:
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded to: {output_path}")

        # Check file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")

        # Extract if it's a zip file
        if output_path.endswith('.zip'):
            print(f"\n📂 Extracting archive...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(FAKEDDIT_DIR)
            print(f"✅ Extracted to: {FAKEDDIT_DIR}")

            # List extracted files
            print(f"\n📋 Extracted files:")
            for item in os.listdir(FAKEDDIT_DIR):
                if item != 'images' and not item.endswith('.zip'):
                    item_path = os.path.join(FAKEDDIT_DIR, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path) / (1024 * 1024)
                        print(f"   - {item} ({size:.2f} MB)")

        # Check for tar.gz
        elif output_path.endswith('.tar.gz') or output_path.endswith('.tgz'):
            print(f"Extracting tar archive...")
            with tarfile.open(output_path, 'r:gz') as tar_ref:
                tar_ref.extractall(FAKEDDIT_DIR)
            print(f"Extracted to: {FAKEDDIT_DIR}")

        return True

    except Exception as e:
        print(f" Error downloading text test set: {e}")
      
        return False


def download_image_test_set():


    url = f"https://drive.google.com/uc?id={IMAGE_TEST_FILE_ID}"
    output_path = os.path.join(FAKEDDIT_DIR, "image_test.zip")

    print(f"\n📥 Downloading from Google Drive...")
    print(f"   File ID: {IMAGE_TEST_FILE_ID}")
    print(f"⚠️  This may take several minutes depending on file size...")

    try:
        gdown.download(url, output_path, quiet=False)

        # Check file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")

        # Extract if it's a zip file
        if output_path.endswith('.zip'):
            print(f"\n📂 Extracting images...")
            print(f"   This may take a while for large image archives...")

            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                # Extract to images directory
                zip_ref.extractall(IMAGES_DIR)

            print(f"Extracted to: {IMAGES_DIR}")

            # Count images
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
            image_count = sum(1 for f in Path(IMAGES_DIR).rglob('*')
                            if f.suffix.lower() in image_extensions)
            print(f"   Total images: {image_count}")

        # Check for tar.gz
        elif output_path.endswith('.tar.gz') or output_path.endswith('.tgz'):
            print(f"\n📂 Extracting tar archive...")
            with tarfile.open(output_path, 'r:gz') as tar_ref:
                tar_ref.extractall(IMAGES_DIR)
            print(f"Extracted to: {IMAGES_DIR}")

        return True

    except Exception as e:
        print(f" Error downloading image test set: {e}")
        return False


def verify_downloads():
    # Check for text files (TSV, CSV, etc.)
    text_files = [f for f in os.listdir(FAKEDDIT_DIR)
                  if f.endswith(('.tsv', '.csv', '.txt')) and os.path.isfile(os.path.join(FAKEDDIT_DIR, f))]

    print(f"Text files found: {len(text_files)}")
    for file in text_files:
        path = os.path.join(FAKEDDIT_DIR, file)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   - {file} ({size_mb:.2f} MB)")

    # Check for images
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = list(Path(IMAGES_DIR).rglob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]

    print(f"Images found: {len(image_files)}")

    if len(image_files) > 0:
        # Show some sample image paths
        print(f"\n   Sample image paths:")
        for img in list(image_files)[:3]:
            print(f"   - {img.name}")

    # Check if we have test files
    test_files = [f for f in text_files if 'test' in f.lower()]
    if test_files:
        print(f"Test files detected:")
        for f in test_files:
            print(f"   - {f}")

    if len(text_files) > 0 and len(image_files) > 0:
        return True
    else:
        print("INCOMPLETE DOWNLOAD")
        if len(text_files) == 0:
            print("No text files found")
        if len(image_files) == 0:
            print("No images found")
        return False


def show_text_file_preview():
    """Show preview of downloaded text files"""
  
    # Find TSV or CSV files
    text_files = [f for f in os.listdir(FAKEDDIT_DIR)
                  if f.endswith(('.tsv', '.csv')) and os.path.isfile(os.path.join(FAKEDDIT_DIR, f))]

    if not text_files:
        print("No TSV/CSV files found")
        return

    # Preview first file
    import pandas as pd

    file_path = os.path.join(FAKEDDIT_DIR, text_files[0])
    print(f"Previewing: {text_files[0]}")

    try:
        # Try reading as TSV first
        df = pd.read_csv(file_path, sep='\t', nrows=5)
      
        display_cols = [c for c in df.columns[:5]]
        print(df[display_cols].head(3).to_string())
    except Exception as e:
        # Try as CSV
        try:
            df = pd.read_csv(file_path, nrows=5)
            display_cols = [c for c in df.columns[:5]]
            print(df[display_cols].head(3).to_string())
        except Exception as e2:
            print(f"Error reading file: {e2}")


def organize_files():
    """Organize downloaded files into proper structure"""


    # Look for test files and rename them appropriately
    for file in os.listdir(FAKEDDIT_DIR):
        if os.path.isfile(os.path.join(FAKEDDIT_DIR, file)):
            if 'test' in file.lower() and file.endswith(('.tsv', '.csv')):
                # Rename to test.tsv
                old_path = os.path.join(FAKEDDIT_DIR, file)
                new_path = os.path.join(FAKEDDIT_DIR, 'test.tsv')

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed {file} -> test.tsv")

    # Check if images are in subdirectories and move them up
    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                source = os.path.join(root, file)
                dest = os.path.join(IMAGES_DIR, file)

                # Only move if it's in a subdirectory
                if root != IMAGES_DIR and not os.path.exists(dest):
                    os.rename(source, dest)

    print("Files organized!")



def download_all():

    # Install gdown
    install_gdown()

    # Download text test set
    text_success = download_text_test_set()

    # Download image test set
    image_success = download_image_test_set()

    # Organize files
    if text_success or image_success:
        organize_files()

    # Verify everything
    verify_downloads()

    # Show preview
    show_text_file_preview()



if __name__ == "__main__":
    download_all()