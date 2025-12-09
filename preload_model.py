# Preload easyocr models for faster container runtime
import easyocr

easyocr.Reader(['en'])
