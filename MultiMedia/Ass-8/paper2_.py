import cv2
import numpy as np
import heapq
from collections import defaultdict
import time
import os
import pandas as pd
import numpy as np
import cv2
import pandas as pd
import time
import pickle

def burrows_wheeler_transform(data):
    n = len(data)
    table = [data[i:] + data[:i] for i in range(n)]
    table.sort()
    return ''.join(row[-1] for row in table), table.index(data)

def inverse_bwt(r, index):
    n = len(r)
    table = [''] * n
    for _ in range(n):
        table = sorted([r[i] + table[i] for i in range(n)])
    return table[index]

def run_length_encode(data):
    encoded = []
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i + 1]:
            i += 1
            count += 1
        encoded.append((data[i], count))
        i += 1
    return encoded

def run_length_decode(encoded):
    return ''.join(char * count for char, count in encoded)

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = defaultdict(int)
    for char in data:
        frequency[char] += 1
    
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left, merged.right = left, right
        heapq.heappush(heap, merged)
    
    return heap[0]

def build_huffman_codes(node, prefix='', codebook={}):
    if node:
        if node.char is not None:
            codebook[node.char] = prefix
        build_huffman_codes(node.left, prefix + '0', codebook)
        build_huffman_codes(node.right, prefix + '1', codebook)
    return codebook

def huffman_encode(data):
    tree = build_huffman_tree(data)
    codes = build_huffman_codes(tree)
    encoded_data = ''.join(codes[char] for char in data)
    return encoded_data, codes

def huffman_decode(encoded_data, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    
    current_code = ''
    decoded_output = ''
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_output += reverse_codes[current_code]
            current_code = ''
    
    return decoded_output

def compress_image(image_path, mode='rgb'):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR if mode == 'rgb' else cv2.IMREAD_GRAYSCALE)
    
    if mode == 'bw':
        image = (image > 128).astype(np.uint8)
    
    compressed_data = []
    shape = image.shape
    channels = 1 if mode in ['gray', 'bw'] else image.shape[-1]
    
    for channel in range(channels):
        data = image[:, :, channel].flatten() if mode == 'rgb' else image.flatten()
        
        bwt_data, index = burrows_wheeler_transform(''.join(map(chr, data)))
        rle_data = run_length_encode(bwt_data)
        encoded_huffman, codes = huffman_encode(''.join(char for char, _ in rle_data))
        
        compressed_data.append((encoded_huffman, codes, index, [count for _, count in rle_data]))
    
    return compressed_data, shape

def decompress_image(compressed_data, shape, mode='rgb'):
    decompressed_channels = []
    channels = 1 if mode in ['gray', 'bw'] else shape[-1]
    
    for encoded_huffman, codes, index, rle_counts in compressed_data:
        decoded_huffman = huffman_decode(encoded_huffman, codes)
        rle_data = [(decoded_huffman[i], rle_counts[i]) for i in range(len(rle_counts))]
        bwt_decoded = run_length_decode(rle_data)
        original_data = inverse_bwt(bwt_decoded, index)
        decompressed_channels.append(np.array([ord(c) for c in original_data], dtype=np.uint8))
    
    if mode in ['gray', 'bw']:
        image = decompressed_channels[0].reshape(shape)
    else:
        image = np.stack(decompressed_channels, axis=-1).reshape(shape)
    
    return image
    

def process_images(image_paths):
    results = []
    modes = ['bw', 'bw', 'gray', 'gray', 'rgb', 'rgb']
    
    for i, image_path in enumerate(image_paths):
        mode = modes[i]
        original_size = os.path.getsize(image_path)  # Original file size in bytes
        
        start_time = time.time()
        compressed_data, shape = compress_image(image_path, mode)
        compression_time = (time.time() - start_time) * 1000  # Convert to ms
        
        compressed_size = sum(len(encoded_huffman) for encoded_huffman, _, _, _ in compressed_data) // 8  # Convert bits to bytes
        compression_ratio = (compressed_size / original_size) * 100
        space_saving = 100 - compression_ratio
        
        results.append([
            image_path, original_size, compressed_size, compression_ratio, space_saving, compression_time
        ])
    
    df = pd.DataFrame(results, columns=[
        'Image Path', 'Original Size (Bytes)', 'Compressed Size (Bytes)', 'Compression Ratio (%)', 'Space Saving (%)', 'Compression Time (ms)'
    ])
    return df
    
image_paths = ['./data/b&w1_mod.png',
              './data/b&w2_mod.png',
              './data/gray1_mod.png',
              './data/gray2_mod.png',
              './data/rgb1_mod.png',
              './data/rgb2_mod.png']
df2 = process_images(image_paths)
print(df2)
df2.to_csv("paper2.csv",index=False)