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
import logging

# supress ocr noise
logging.getLogger('easyocr').setLevel(logging.ERROR)

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

__version__ = '1.1.2'

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
parser.add_argument('-o', '--textOutputDir', default="text_data", type=str,
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
    reader = easyocr.Reader(['en'],
                            model_storage_directory='/opt/easyocr',
                            download_enabled=False,
                            gpu=options.useGpu,
                            quantize=True,
                            verbose=False)  # ['en', 'fr', 'de', ...]

    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/*.{options.fileFilter}",fail_if_empty=False)
    logger.info(f"Total no. of file(s) found: {len(mapper)} ")
    for input_file, output_file in mapper:
        # Read each input file from the input directory that matches the input filter specified
        dcm_img = read_input_dicom(input_file)

        # check if a valid image file is returned
        if dcm_img is None:
            continue

        # locate text in pixel data
        extracted_text = extract_text_from_image(reader, dcm_img)

        save_as_text_file(extracted_text, options.textOutputDir, output_file)

        # Save the file in o/p directory in the specified o/p type
        if options.outputType == "dcm":
            save_dicom(dcm_img, output_file)
        else:
            save_as_image(dcm_img, output_file, options.outputType)


def extract_text_from_image(reader, ds) -> str:
    rgb_img = convert_image(ds.pixel_array)

    # Run OCR
    extracted_text = reader.readtext(rgb_img,
                                     detail=0,
                                     paragraph=False,
                                     contrast_ths=0.05,  # lower makes it more sensitive
                                     adjust_contrast=0.7,  # 0.5–0.9 is usually ideal
                                     text_threshold=0.4)

    if extracted_text:
        logger.info(f"Extracted text: {extracted_text}")

    return extracted_text
        
def read_input_dicom(input_file_path):
    """
    1) Read an input dicom file
    """
    try:
        logger.info(f"Reading input file : {input_file_path.name}")
        ds = pydicom.dcmread(str(input_file_path))
        if 'PixelData' not in ds:
            logger.warning("No pixel data in this DICOM.")
            return None
    except Exception as ex:
        logger.error(f"unable to read dicom file: {ex}")
        return None

    return ds

def save_as_text_file(text, op_dir, output_file_path):
    text_file_path = Path(output_file_path.parent / op_dir / f"{output_file_path.name.replace('.dcm', '')}.txt")
    # Create parent directories if needed
    text_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing text file: {text_file_path.name}")
    with open(text_file_path, "w") as text_file:
        text_file.write(" ".join(text))

def convert_image(img):
    img = img.astype(np.float32)

    # Replace NaN / Inf
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
        img = img * 255.0
    else:
        # Flat image → zero image
        img = np.zeros_like(img)

    img = img.astype(np.uint8)

    # Grayscale → RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Ensure contiguous memory (IMPORTANT for EasyOCR)
    img = np.ascontiguousarray(img)

    return img


def save_dicom(dicom_file, output_path):
    """
    Save a dicom file to an output path
    """
    logger.info(f"Saving dicom file: {output_path.name}")
    dicom_file.save_as(str(output_path))

def save_as_image(dcm_file, output_file_path, file_ext):
    """
    Save the pixel array of a dicom file as an image file
    """
    pixel_array_numpy = dcm_file.pixel_array
    output_file_path = str(output_file_path).replace('dcm', file_ext)
    logger.info(f"Saving output file as: [{output_file_path}]")
    logger.debug(f"Photometric Interpretation is {dcm_file.PhotometricInterpretation}")

    # Prevents color inversion happening while saving as images
    if 'YBR' in dcm_file.PhotometricInterpretation:
        logger.debug(f"Explicitly converting color space to RGB")
        pixel_array_numpy = convert_color_space(pixel_array_numpy, "YBR_FULL", "RGB")

    cv2.imwrite(output_file_path,cv2.cvtColor(pixel_array_numpy,cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
