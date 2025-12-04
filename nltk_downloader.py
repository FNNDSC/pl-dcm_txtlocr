# nltk_downloader.py
import nltk
import easyocr

nltk.download('punkt', download_dir='/usr/local/lib/nltk_data')
nltk.download('punkt_tab', download_dir='/usr/local/lib/nltk_data')
nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/local/lib/nltk_data')
nltk.download('maxent_ne_chunker_tab', download_dir='/usr/local/lib/nltk_data')
nltk.download('maxent_ne_chunker', download_dir='/usr/local/lib/nltk_data')
nltk.download('words', download_dir='/usr/local/lib/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/lib/nltk_data')
easyocr.Reader(['en'], gpu=False)
