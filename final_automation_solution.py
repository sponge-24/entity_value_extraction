import os
import pandas as pd
import re
import easyocr
import spacy
import csv
import sys
import time

# Load spaCy NER model for Quantity recognition
nlp = spacy.load("./output/model-best")  # Custom-trained NER model

# Conversion dictionary for full-form units
conversion_dict = {
    'width': {'cm': 'centimetre', 'ft': 'foot', 'in': 'inch', 'm': 'metre', 'mm': 'millimetre', 'yd': 'yard'},
    'depth': {'cm': 'centimetre', 'ft': 'foot', 'in': 'inch', 'm': 'metre', 'mm': 'millimetre', 'yd': 'yard'},
    'height': {'cm': 'centimetre', 'ft': 'foot', 'in': 'inch', 'm': 'metre', 'mm': 'millimetre', 'yd': 'yard'},
    'item_weight': {'g': 'gram', 'kg': 'kilogram', 'µg': 'microgram', 'mg': 'milligram', 'oz': 'ounce', 'lb': 'pound', 't': 'ton'},
    'maximum_weight_recommendation': {'g': 'gram', 'kg': 'kilogram', 'µg': 'microgram', 'mg': 'milligram', 'oz': 'ounce', 'lb': 'pound', 't': 'ton'},
    'voltage': {'kv': 'kilovolt', 'mv': 'millivolt', 'v': 'volt'},
    'wattage': {'kw': 'kilowatt', 'w': 'watt'},
    'item_volume': {'cl': 'centilitre', 'cu ft': 'cubic foot', 'cu in': 'cubic inch', 'cup': 'cup', 'dl': 'decilitre', 'fl oz': 'fluid ounce', 'gallon': 'gallon', 'l': 'litre', 'ml': 'millilitre', 'oz': 'ounce', 'pint': 'pint', 'qt': 'quart'}
}

# Cleaner Principle - Units for extraction and processing
units = {
    "width": ["cm", "ft", "in", "m", "mm", "yd"],
    "depth": ["cm", "ft", "in", "m", "mm", "yd"],
    "height": ["cm", "ft", "in", "m", "mm", "yd"],
    "item_weight": ["g", "kg", "µg", "mg", "oz", "lb", "t"],
    "maximum_weight_recommendation": ["g", "kg", "µg", "mg", "oz", "lb", "t"],
    "voltage": ["kv", "mv", "v"],
    "wattage": ["kw", "w"],
    "item_volume": ["cl", "cu ft", "cu in", "cup", "dl", "fl oz", "gallon", "l", "ml", "oz", "pint", "qt"]
}
all_units = [unit.lower() for sublist in units.values() for unit in sublist]

# EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to extract image name
def get_image_name(image_link):
    return os.path.basename(image_link)

# Function to convert abbreviations to full forms
def convert_units(entity_name, text):
    if entity_name in conversion_dict:
        for abbrev, full_form in conversion_dict[entity_name].items():
            text = re.sub(rf"\b{abbrev}\b", full_form, text)
    return text

# Function to clean and extract measurements from text using regex
def extract_measurements(text):
    text = text.lower()  # Handle varying capitalizations
    pattern = r'(\d+(\.\d+)?)\s*({})'.format('|'.join(all_units))  # Pattern for numbers followed by units
    matches = re.findall(pattern, text)
    return [(match[0], match[2]) for match in matches]  # Return list of (value, unit) tuples

# Function to filter values based on the entity name
def get_value_for_entity(entity_name, measurements):
    if entity_name in units:
        valid_units = units[entity_name]
        for value, unit in measurements:
            if unit in valid_units:
                return f"{value} {unit}"
    return ""  # Return "0, empty string" if no valid measurement is found

# Function to process each image and extract the necessary quantity
def process_image(image_link, entity_name):
    image_name = get_image_name(image_link)
    image_path = os.path.join('./test_preprocessed', image_name)  # Assuming images are in 'images/' folder

    # Step 1: Extract text using EasyOCR
    result = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(result)

    # Step 2: Perform NER with spaCy
    doc = nlp(extracted_text)
    quantities = [ent.text for ent in doc.ents if ent.label_ == "QUANTITY"]

    # Step 3: Clean the extracted text using the cleaner principle
    if quantities:
        measurements = extract_measurements(" ".join(quantities))
        # Step 4: Filter the extracted measurements for the given entity
        entity_value = get_value_for_entity(entity_name, measurements)

        if entity_value != "":
            # Step 5: Convert the abbreviation to full form
            entity_value = convert_units(entity_name, entity_value)
        return entity_value

    return ""

# Checkpoint logic
def save_checkpoint(last_index):
    with open('checkpoint.txt', 'w') as f:
        f.write(str(last_index))

def load_checkpoint():
    if os.path.exists('checkpoint.txt'):
        with open('checkpoint.txt', 'r') as f:
            return int(f.read().strip())
    return 0  # If no checkpoint file exists, start from the beginning

# Process all rows in CSV and collect results
def process_csv(input_file, output_file, batch_size=1000):

    df = pd.read_csv(input_file)

    # Load checkpoint (if exists)
    start_index = load_checkpoint()

    # Open the output CSV in append mode and skip writing header if file exists
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='', encoding='utf-8') as outfile:
        fieldnames = ['index', 'prediction']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write the header only if the file doesn't already exist
        if not file_exists:
            writer.writeheader()

        # Process rows starting from the checkpoint
        for index, row in df.iterrows():
            if index < start_index:
                continue  # Skip already processed rows

            image_link = row['image_link']
            entity_name = row['entity_name']
            index_value = row['index']

            # Extract and process image to get prediction
            prediction = process_image(image_link, entity_name)

            # Write result to the CSV file after processing each image
            writer.writerow({'index': index_value, 'prediction': prediction})

            print(f"Completed processing image {index_value}: {image_link}")

            # Flush the CSV file to ensure it's written to disk
            outfile.flush()

            # Save checkpoint every `batch_size` rows
            if (index + 1) % batch_size == 0:
                save_checkpoint(index + 1)
                print(f"Processed {index + 1} rows. Restarting... Last processed index saved.")
                restart_program()

    print(f"Processing completed. Predictions saved to {output_file}.")

# Function to restart the script
def restart_program():
    """
    This function schedules a restart of the current Python program.
    """
    time.sleep(30)
    python = sys.executable
    os.execl(python, python, *sys.argv)  # Restart the script using current arguments

# File paths
input_csv = './amazon_ml/student_resource 3/dataset/test.csv'  # Your input CSV with image links
output_csv = './new_test_out_final_final.csv'

# Run the processing with batch size of 500 rows
process_csv(input_csv, output_csv, batch_size=1000)
