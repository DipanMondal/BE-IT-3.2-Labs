import numpy as np
import cv2
import pandas as pd
import time
import multiprocessing
import pickle

def bitwise_rle_encode(data):
    encoded = []
    count = 1
    prev = data[0]
    for i in range(1, len(data)):
        if data[i] == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = data[i]
            count = 1
    encoded.append((prev, count))  # Add last sequence
    return encoded
    
def save_encoded(filename, encoded_data):
    """Save encoded data using pickle."""
    with open(filename, "wb") as file:
        pickle.dump(encoded_data, file)
    
def image_to_bitwise_rle(image_path, mode='grayscale'):
    start_time = time.time()
    
    if mode == 'bw':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.where(image > 127, 255, 0).astype(np.uint8)  # Convert to B/W
    elif mode == 'grayscale':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif mode == 'rgb':
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Invalid mode. Choose from 'bw', 'grayscale', 'rgb'")
    
    original_size = image.size  # Total pixels
    if mode == 'rgb':
        flattened = image.reshape(-1, 3)  # Flatten RGB
        with multiprocessing.Pool(3) as pool:
            encoded = pool.map(bitwise_rle_encode, [flattened[:, i] for i in range(3)])
    else:
        flattened = image.flatten()
        encoded = bitwise_rle_encode(flattened)
        
    save_encoded(image_path+"_paper2_encodes.pkl",encoded)
    
    compressed_size = sum(len(enc) for enc in encoded) * 2  # Each (value, count) pair
    compression_ratio = (compressed_size / original_size) * 100
    space_saving = 100 - compression_ratio
    compression_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return original_size, compressed_size, compression_ratio, space_saving, compression_time

def evaluate_bitwise_images(image_paths, modes=['bw', 'grayscale', 'rgb']):
    results = []
    for i in range(len(image_paths)):
        image_path=image_paths[i]
        mode=modes[i//2]
        orig_size, comp_size, comp_ratio, space_save, comp_time = image_to_bitwise_rle(image_path, mode)
        results.append([image_path, mode, orig_size, comp_size, comp_ratio, space_save, comp_time])
    
    df = pd.DataFrame(results, columns=['Image', 'Mode', 'Original Size', 'Compressed Size', 'Compression Ratio (%)', 'Space Saving (%)', 'Compression Time (ms)'])
    return df
    
if __name__ == "__main__":
    image_paths = ['./data/b&w1_mod.png',
              './data/b&w2_mod.png',
              './data/gray1_mod.png',
              './data/gray2_mod.png',
              './data/rgb1_mod.png',
              './data/rgb2_mod.png']
    df_results = evaluate_bitwise_images(image_paths)
    print(df_results)
    df_results.to_csv("paper2.csv",index=False)