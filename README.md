# Entity Value Extraction from Product Images

This project processes product images to extract specific entity values (like width, height, weight, volume, etc.) using a combination of Optical Character Recognition (OCR) and a custom-trained Named Entity Recognition (NER) model. The extracted values are saved into a CSV file.

## Features

- **Image Download**: Download images from a given CSV file containing image URLs.
- **Image Preprocessing**: Apply contrast enhancement and noise removal using `PIL` and `OpenCV`.
- **NER Model**: Extract entity values using a custom-trained spaCy NER model for entity recognition.
- **OCR**: Use EasyOCR to extract text from product images.
- **Multiprocessing**: Speed up image processing using parallel workers.
- **Batch Processing**: Handle large datasets efficiently, including a checkpoint system for resuming the process after interruptions.

## Technologies

- Python 3.8+
- OpenCV
- PIL (Pillow)
- spaCy
- EasyOCR
- Multiprocessing
- pandas
- tqdm (progress bar)
