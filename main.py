import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Set
from dataclasses import dataclass
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

@dataclass
class ImageFile:
    """Represents an image file with its properties."""
    path: Path
    original_name: str
    extension: str

class ImageRenamer:
    """Handles the renaming of image files based on their content."""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.PNG', 'JPG'}
    MAX_FILENAME_LENGTH = 255
    
    def __init__(self, directory: str, skip_existing: bool = True, verbose = False):
        """
        Initialize the ImageRenamer.
        
        Args:
            directory: Directory containing the images
            skip_existing: Whether to skip files that already start with '_'
        """
        self.directory = Path(directory)
        self.skip_existing = skip_existing
        self.verbose = verbose
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure logging for the application."""
        setlevel = None
        if self.verbose:
            setlevel = logging.INFO
        else:
            setlevel = logging.WARN

        logging.basicConfig(
            level=setlevel,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_renamer.log'),
                logging.StreamHandler()
            ]
        )
    
    def get_image_files(self) -> List[ImageFile]:
        """Get all valid image files from the directory."""
        image_files = []
        
        for file_path in self.directory.iterdir():
            if not self._is_valid_image_file(file_path):
                continue
                
            image_files.append(ImageFile(
                path=file_path,
                original_name=file_path.stem,
                extension=file_path.suffix.lower()
            ))
            
        return image_files
    
    def _is_valid_image_file(self, file_path: Path) -> bool:
        """Check if the file is a valid image file to process."""
        if not file_path.is_file():
            return False
        
        if self.skip_existing and file_path.stem.startswith('_'):
            return False
            
        return file_path.suffix.lower() in self.VALID_EXTENSIONS
    
    def extract_text_from_image(self, image_path: Path) -> Optional[str]:
        """Extract text from an image using OCR."""
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, lang='eng')
                return text.strip() if text else None
        except Exception as e:
            logging.error(f"Error extracting text from {image_path}: {e}")
            return None
    
    def clean_text_for_filename(self, text: str, max_length: int) -> str:
        """Clean and format text for use as a filename."""
        # Replace special characters and whitespace
        text = re.sub(r"[^\w\s]", "_", text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"_+", "_", text)
        
        # Ensure the text starts with a letter or number
        text = re.sub(r"^[^a-zA-Z0-9]+", "", text)
        
        # Truncate if necessary
        return text[:max_length].rstrip('_')
    
    def generate_unique_filename(self, base_name: str, extension: str, existing_names: Set[str]) -> str:
        """Generate a unique filename by adding a number if necessary."""
        counter = 0
        new_name = f"_{base_name}{extension}"
        
        while new_name in existing_names:
            counter += 1
            truncated_base = base_name[:self.MAX_FILENAME_LENGTH - len(str(counter)) - len(extension) - 2]
            new_name = f"_{truncated_base}_{counter}{extension}"
            
        return new_name
    
    def process_image(self, image_file: ImageFile, existing_names: Set[str]) -> None:
        """Process a single image file."""
        try:
            # Extract text from image
            text = self.extract_text_from_image(image_file.path)
            
            if not text:
                # Use a generic name if no text was extracted
                base_name = f"meme_{len(existing_names)}"
            else:
                # Clean and prepare the extracted text
                base_name = self.clean_text_for_filename(
                    text,
                    self.MAX_FILENAME_LENGTH - len(image_file.extension) - 2
                )
            
            # Generate unique filename
            new_name = self.generate_unique_filename(base_name, image_file.extension, existing_names)
            new_path = image_file.path.parent / new_name
            
            # Rename the file
            image_file.path.rename(new_path)
            existing_names.add(new_name)
            
            logging.info(f"Renamed: {image_file.original_name} -> {new_name}")
            
        except Exception as e:
            logging.error(f"Error processing {image_file.path}: {e}")
    
    def process_directory(self) -> None:
        """Process all images in the directory."""
        image_files = self.get_image_files()
        if not image_files:
            logging.info("No valid image files found to process")
            return
            
        existing_names = {f.name for f in self.directory.iterdir()}
        
        logging.info(f"Processing {len(image_files)} images...")
        
        # Process images in parallel with a progress bar
        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(
                    lambda x: self.process_image(x, existing_names),
                    image_files
                ),
                total=len(image_files),
                desc="Renaming images"
            ))

def main() -> None:
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rename image files based on their content using OCR")
    parser.add_argument("directory", help="Directory containing the images")
    parser.add_argument("--process-all", action="store_true", dest="skip_existing",
                       help="Process all files, including those starting with '_'")
    parser.add_argument("--verbose", action="store_true", dest="verbose",
                       help="All verbosity or nothing?")
    
    args = parser.parse_args()
    
    renamer = ImageRenamer(args.directory, skip_existing=args.skip_existing, verbose=args.verbose)
    renamer.process_directory()

if __name__ == "__main__":
    main()
    