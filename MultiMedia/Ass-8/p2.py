from PIL import Image
import numpy as np

def prepare_image(image_path, image_type):
    """
    Convert an image to the desired format: black & white, grayscale, or RGB.
    
    Parameters:
        image_path (str): Path to the input image (e.g., PNG).
        image_type (str): One of 'bw', 'gray', or 'rgb'.
        
    Returns:
        data (np.ndarray): Processed image data as a flat byte array.
        shape (tuple): Original image shape.
        mode (str): Mode after conversion.
    """
    image = Image.open(image_path)

    if image_type == 'bw':
        # Convert to grayscale then threshold to black and white (1-bit)
        gray = image.convert('L')
        bw = gray.point(lambda x: 0 if x < 128 else 255, '1')  # 1-bit per pixel
        data = np.packbits(np.array(bw).astype(np.uint8))  # Compress bits into bytes
        mode = '1'
        shape = bw.size  # width, height

    elif image_type == 'gray':
        gray = image.convert('L')  # 8-bit grayscale
        data = np.array(gray).astype(np.uint8).flatten()
        mode = 'L'
        shape = gray.size

    elif image_type == 'rgb':
        rgb = image.convert('RGB')  # Ensure it's in RGB
        data = np.array(rgb).astype(np.uint8).flatten()
        mode = 'RGB'
        shape = rgb.size

    else:
        raise ValueError("image_type must be one of: 'bw', 'gray', 'rgb'")

    return data, shape, mode


import numpy as np
from collections import Counter, defaultdict
from bitarray import bitarray
import heapq

# Step 1: Burrows-Wheeler Transform (Simplified for small data)
def bwst_transform(data: bytes) -> bytes:
    n = len(data)
    rotations = [data[i:] + data[:i] for i in range(n)]
    sorted_rotations = sorted(rotations)
    return bytes([rot[-1] for rot in sorted_rotations])

# Step 2: Dynamic Byte Remapping
def remap_bytes(data: bytes):
    freq = Counter(data)
    sorted_bytes = [b for b, _ in freq.most_common()]
    byte_map = {old: i for i, old in enumerate(sorted_bytes)}
    remapped = bytes([byte_map[b] for b in data])
    return remapped, byte_map

# Step 3: Vertical Byte Reading (bit-plane separation)
def vertical_bit_reading(data: bytes) -> bitarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    reshaped = bits.reshape(-1, 8).T  # 8 rows for 8 bit-planes
    return bitarray(reshaped.flatten().tolist())

# Step 4: Bit-level RLE
def bit_rle(bits: bitarray):
    rle = []
    current = bits[0]
    count = 1
    for bit in bits[1:]:
        if bit == current:
            count += 1
        else:
            rle.append((current, count))
            current = bit
            count = 1
    rle.append((current, count))
    return rle

# Step 5: Huffman Encoding for RLE runs
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq_map.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        new = HuffmanNode(freq=left.freq + right.freq)
        new.left = left
        new.right = right
        heapq.heappush(heap, new)
    return heap[0]

def generate_huffman_codes(node, prefix="", codebook={}):
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    generate_huffman_codes(node.left, prefix + "0", codebook)
    generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(rle_data):
    run_lengths = [length for _, length in rle_data]
    freq_map = Counter(run_lengths)
    tree = build_huffman_tree(freq_map)
    codes = generate_huffman_codes(tree)
    encoded_bits = bitarray()
    for bit, length in rle_data:
        encoded_bits.extend([bit])  # bit itself
        encoded_bits.extend(codes[length])  # Huffman code for run length
    return encoded_bits, codes

# Wrapper Function for Full Compression
def improved_rle_compression(data: bytes):
    step1 = bwst_transform(data)
    step2, byte_map = remap_bytes(step1)
    step3 = vertical_bit_reading(step2)
    step4 = bit_rle(step3)
    step5, huff_codes = huffman_encode(step4)
    return step5, {
        "original_size": len(data),
        "compressed_bits": len(step5),
        "byte_map": byte_map,
        "huffman_codes": huff_codes
    }


if __name__=="__main__":
	from PIL import Image
	from io import BytesIO

	# Example: Use the prepare_image function from earlier
	data, shape, mode = prepare_image("./data/rgb1_mod.png", "gray")

	# Run Improved RLE Compression
	compressed_bits, info = improved_rle_compression(data.tobytes())

	# Print Summary
	print(f"Original Size: {info['original_size']} bytes")
	print(f"Compressed Size: {info['compressed_bits'] / 8:.2f} bytes")
	print(f"Compression Ratio: {info['original_size'] / (info['compressed_bits'] / 8):.2f}x")
