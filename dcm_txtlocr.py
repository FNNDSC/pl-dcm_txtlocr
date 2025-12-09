#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin, PathMapper
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from datetime import datetime
from PIL import Image
from difflib import SequenceMatcher
from loguru import logger
import numpy as np
import pydicom
import re
import sys
import easyocr
import cv2

LOG = logger.debug

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> │ "
    "<level>{level: <5}</level> │ "
    "<yellow>{name: >28}</yellow>::"
    "<cyan>{function: <30}</cyan> @"
    "<cyan>{line: <4}</cyan> ║ "
    "<level>{message}</level>"
)
logger.remove()
logger.add(sys.stderr, format=logger_format)

__version__ = '1.0.6'

DISPLAY_TITLE = r"""


         888             888                                888             888    888                          
         888             888                                888             888    888                          
         888             888                                888             888    888                          
88888b.  888         .d88888  .d8888b 88888b.d88b.          888888 888  888 888888 888  .d88b.   .d8888b 888d888
888 "88b 888        d88" 888 d88P"    888 "888 "88b         888    `Y8bd8P' 888    888 d88""88b d88P"    888P"  
888  888 888 888888 888  888 888      888  888  888         888      X88K   888    888 888  888 888      888    
888 d88P 888        Y88b 888 Y88b.    888  888  888         Y88b.  .d8""8b. Y88b.  888 Y88..88P Y88b.    888    
88888P"  888         "Y88888  "Y8888P 888  888  888 88888888 "Y888 888  888  "Y888 888  "Y88P"   "Y8888P 888    
888                                                                                                             
888                                                                                                             
888                                                                                                                                         
""" + "\t\t -- version " + __version__ + " --\n\n"


parser = ArgumentParser(description='A ChRIS plugin to locate text in a DICOM file',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--inspectTags',nargs="?", default=None, const="", type=str,
                    help='Comma separated DICOM tags')
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
parser.add_argument('-f', '--fileFilter', default='dcm', type=str,
                    help='input file filter glob')
parser.add_argument('-t', '--outputType', default='dcm', type=str,
                    help='output file type(extension only)')
parser.add_argument('-u', '--useGpu', default=False, action="store_true",
                    help='If specified, use available gpu')


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='A ChRIS plugin to detect text in a DICOM file',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='2000Mi',    # supported units: Mi, Gi
    min_cpu_limit='4000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)
    # Create a reader for specific languages
    reader = easyocr.Reader(['en'], gpu=options.useGpu, quantize=True)  # ['en', 'fr', 'de', ...]

    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*.{options.fileFilter}",fail_if_empty=False)
    print(f"Total no. of file(s) found: {len(mapper)} ")
    for input_file, output_file in mapper:
        # Read each input file from the input directory that matches the input filter specified
        dcm_img = read_input_dicom(input_file, reader, options.inspectTags)

        # check if a valid image file is returned
        if dcm_img is None:
            continue

        # Save the file in o/p directory in the specified o/p type\
        if options.outputType == "dcm":
            save_dicom(dcm_img, output_file)
        else:
            save_as_image(dcm_img, output_file, options.outputType)
        print("\n\n")
        
def read_input_dicom(input_file_path, reader, tags):
    """
    1) Read an input dicom file
    """
    ds = None
    try:
        print(f"Reading input file : {input_file_path.name}")
        ds = pydicom.dcmread(str(input_file_path))
        if 'PixelData' not in ds:
            print("No pixel data in this DICOM.")
            return None
    except Exception as ex:
        print(f"unable to read dicom file: {ex} \n")
        return None

    # Run OCR
    rgb_img = convert_image(ds.pixel_array)
    extracted_text = reader.readtext(rgb_img,
                                     detail=0,
                                     paragraph=False,
                                     contrast_ths=0.05,       # lower makes it more sensitive
                                     adjust_contrast=0.7,     # 0.5–0.9 is usually ideal
                                     text_threshold=0.4)

    """
        Decide whether to keep or filter a DICOM dataset.

        Logic:
        1) If no extracted_text → return ds
        2) If extracted_text exists:
            a) If tag is not None AND detect_phi(tokens, ds, tags) → return None
            b) Else → return ds
        """
    if extracted_text:
        print(f"Extracted Text: {extracted_text}\n")
        return ds
        # if tags is None:
        #    return None

        # tokens = tokenize_strings(extracted_text)

        # if detect_phi(tokens, ds, tags):
        #     return None  # PHI detected
        #return ds  # Text exists but no PHI detected or tag is None

    # No extracted text → keep dataset
    # return ds
    return None

def convert_image(img):
    # Normalize pixel values (optional but recommended for CT/MRI)
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    # If grayscale → convert to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def _dicom_to_image(ds):
    pixel_array = ds.pixel_array  # This is usually a NumPy array

    # Check shape
    print("DICOM pixel_array shape:", pixel_array.shape)

    # Choose the middle slice if it's 3D
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[pixel_array.shape[0] // 2]

    # Normalize and convert to uint8 if necessary
    if pixel_array.dtype != np.uint8:
        pixel_array = (255 * (pixel_array - np.min(pixel_array)) / np.ptp(pixel_array)).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(pixel_array)
    return image

def similarity(a, b):
    """Returns a similarity ratio between 0 and 1."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detect_phi(text, ds, tags, threshold=0.80):
    """
    Detects possible PHI in `text` by comparing it against the extracted
    DICOM text & dates, using exact, substring, and similarity matching.
    """
    all_text_and_dates = extract_text_and_dates(ds, tags)

    flagged = False

    for word in text:  # split input text into tokens
        for dicom_val in all_text_and_dates:

            # --- Exact match ---
            if word.lower() == dicom_val.lower():
                print(f"\n[PHI - EXACT MATCH] '{word}' == '{dicom_val}'")
                flagged = True
                continue


            # --- Similarity (fuzzy) match ---
            score = similarity(word, dicom_val)
            if score >= threshold:
                print(f"\n[PHI - SIMILARITY {score:.2f}] '{word}' ≈ '{dicom_val}'")
                flagged = True
                continue

    return flagged


def save_dicom(dicom_file, output_path):
    """
    Save a dicom file to an output path
    """
    print(f"Saving dicom file: {output_path.name}")
    dicom_file.save_as(str(output_path))

def save_as_image(dcm_file, output_file_path, file_ext):
    """
    Save the pixel array of a dicom file as an image file
    """
    pixel_array_numpy = dcm_file.pixel_array
    output_file_path = str(output_file_path).replace('dcm', file_ext)
    print(f"Saving output file as {output_file_path}")
    print(f"Photometric Interpretation is {dcm_file.PhotometricInterpretation}")

    # Prevents color inversion happening while saving as images
    if 'YBR' in dcm_file.PhotometricInterpretation:
        print(f"Explicitly converting color space to RGB")
        pixel_array_numpy = convert_color_space(pixel_array_numpy, "YBR_FULL", "RGB")

    cv2.imwrite(output_file_path,cv2.cvtColor(pixel_array_numpy,cv2.COLOR_RGB2BGR))


def extract_text_and_dates(ds: Dataset, tags=None):
    """
    Extract text, dates (MM/DD/YYYY), and PN names (First Last) from a DICOM dataset.

    Optional:
        tags (str): comma-separated list of DICOM tags to extract.
                    Supports keywords (e.g. "PatientName") or hex (e.g. "00100010").

    If tags is None → extract from all fields.
    """
    results = set()

    # ---------------------------------------------------------------------
    # Parse user-provided tags
    # ---------------------------------------------------------------------
    allowed_tags = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        allowed_tags = set()

        for t in tag_list:
            # Keyword (e.g., "PatientName")
            if t.isalpha():
                allowed_tags.add(t)

            # Hex tag (e.g., "00100010")
            else:
                try:
                    hex_tag = int(t, 16)
                    allowed_tags.add(hex_tag)
                except Exception:
                    pass

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def convert_dicom_date(d):
        try:
            return datetime.strptime(str(d), "%Y%m%d").strftime("%m/%d/%Y")
        except Exception:
            return None

    def dicom_name_to_first_last(pn_value):
        if not pn_value:
            return ""
        parts = str(pn_value).split("^")
        last = parts[0] if len(parts) > 0 else ""
        first = parts[1] if len(parts) > 1 else ""
        return f"{first} {last}".strip() if first and last else first or last

    # ---------------------------------------------------------------------
    # Processing logic with tag filtering
    # ---------------------------------------------------------------------
    def process_element(elem):
        """Checks tag filter before processing."""
        tag_num = elem.tag
        tag_name = elem.keyword

        # If tag filtering is enabled
        if allowed_tags is not None:
            if (tag_name not in allowed_tags) and (tag_num not in allowed_tags):
                return  # Skip this field

        process_value(elem.VR, elem.value)

    def process_value(vr, value):
        if value is None or value == "":
            return

        # Sequence
        if vr == "SQ" and isinstance(value, Sequence):
            for item in value:
                traverse(item)
            return

        # Multi-value
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            for v in value:
                process_value(vr, v)
            return

        # Person Name
        if vr == "PN":
            name = dicom_name_to_first_last(value)
            if name:
                results.add(name)
            return

        # Date
        if vr == "DA":
            formatted = convert_dicom_date(value)
            if formatted:
                results.add(formatted)
            return

        # Text VRs
        if vr in {"LO", "LT", "SH", "ST", "UT", "AE", "CS", "UC"}:
            results.add(str(value))
            return

        # Fallback string → treat as text
        if isinstance(value, str):
            results.add(value)

    def traverse(dataset: Dataset):
        for elem in dataset:
            process_element(elem)

    traverse(ds)
    return tokenize_strings(results)




def tokenize_strings(strings):
    """
    Tokenizes a list or set of strings into a flat list of words.

    - Lowercases all text
    - Splits on whitespace
    """
    tokens = []

    for s in strings:
        if not s:
            continue

        # Normalize: lowercase and remove non-alphanumeric chars
        cleaned = s.lower()

        # Split into words
        words = cleaned.split()

        # Add to token list
        tokens.extend(words)

    return tokens


if __name__ == '__main__':
    main()
