import sys
import pandas as pd
sys.path.append(r'./amazon_ml/student_resource 3/src')
from utils import download_images

if __name__ == '__main__':
    csv_file = './amazon_ml/student_resource 3/dataset/test.csv'
    df = pd.read_csv(csv_file)
    image_links = df['image_link'].tolist()

    download_folder = "./test"
    download_images(image_links, download_folder, allow_multiprocessing=True)
