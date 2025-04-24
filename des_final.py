# DES Algorithm Implementation in Python Without External Libraries

# Permutation tables
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

FP = [40, 8, 48, 16, 56, 24, 64, 32,
      39, 7, 47, 15, 55, 23, 63, 31,
      38, 6, 46, 14, 54, 22, 62, 30,
      37, 5, 45, 13, 53, 21, 61, 29,
      36, 4, 44, 12, 52, 20, 60, 28,
      35, 3, 43, 11, 51, 19, 59, 27,
      34, 2, 42, 10, 50, 18, 58, 26,
      33, 1, 41, 9, 49, 17, 57, 25]

# Expansion table (E)
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

# Permutation table (P) used in the f-function
P = [16, 7, 20, 21,
     29, 12, 28, 17,
     1, 15, 23, 26,
     5, 18, 31, 10,
     2, 8, 24, 14,
     32, 27, 3, 9,
     19, 13, 30, 6,
     22, 11, 4, 25]

# S-boxes (8 boxes, each 4x16)
S_BOX = [
    # S1
    [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
     [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
     [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
     [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],
    # S2
    [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
     [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
     [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
     [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],
    # S3
    [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
     [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
     [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],
    # S4
    [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
     [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
     [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
     [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],
    # S5
    [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
     [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
     [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
     [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],
    # S6
    [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
     [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
     [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
     [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],
    # S7
    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],
    # S8
    [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
     [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
     [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
     [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]
]

# Permuted Choice 1 (PC-1) for key schedule
PC1 = [57, 49, 41, 33, 25, 17, 9,
       1, 58, 50, 42, 34, 26, 18,
       10, 2, 59, 51, 43, 35, 27,
       19, 11, 3, 60, 52, 44, 36,
       63, 55, 47, 39, 31, 23, 15,
       7, 62, 54, 46, 38, 30, 22,
       14, 6, 61, 53, 45, 37, 29,
       21, 13, 5, 28, 20, 12, 4]

# Permuted Choice 2 (PC-2) for key schedule
PC2 = [14, 17, 11, 24, 1, 5,
       3, 28, 15, 6, 21, 10,
       23, 19, 12, 4, 26, 8,
       16, 7, 27, 20, 13, 2,
       41, 52, 31, 37, 47, 55,
       30, 40, 51, 45, 33, 48,
       44, 49, 39, 56, 34, 53,
       46, 42, 50, 36, 29, 32]

# Number of left shifts per round for key schedule
SHIFT_TABLE = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


# Helper function for permutation:
def permute(block, table):
    """Permute the block according to the given table."""
    return [block[i - 1] for i in table]


# Helper function for left circular shift:
def shift_left(block, num_shifts):
    """Left shift the list (block) by num_shifts."""
    return block[num_shifts:] + block[:num_shifts]


# Helper function for XOR between two lists of bits:
def xor(a, b):
    """Return the XOR of two bit lists."""
    return [i ^ j for i, j in zip(a, b)]


# Convert a hexadecimal string to a binary list (of 0/1)
def hex_to_bin(hex_str):
    bin_str = ""
    for ch in hex_str:
        # Convert each hex digit (base 16) into a 4-bit binary representation.
        bin_str += format(int(ch, 16), '04b')
    return [int(b) for b in bin_str]


# Convert a binary list to a hexadecimal string.
def bin_to_hex(bin_list):
    bin_str = ''.join(str(bit) for bit in bin_list)
    # Split into groups of 4 bits and convert each group to a hex digit.
    hex_str = ""
    for i in range(0, len(bin_str), 4):
        hex_digit = hex(int(bin_str[i:i+4], 2))[2:]
        hex_str += hex_digit
    return hex_str.upper()


# Key schedule generation to produce 16 round keys:
def generate_keys(key):
    """
    key: list of 64 bits
    Returns a list of 16 subkeys, each 48 bits long.
    """
    # Permute the key using PC-1 to get 56-bit key
    key = permute(key, PC1)
    # Split into left (C) and right (D) halves, 28 bits each
    C = key[:28]
    D = key[28:]
    round_keys = []
    for shift in SHIFT_TABLE:
        # Left shift both halves
        C = shift_left(C, shift)
        D = shift_left(D, shift)
        # Combine halves
        combined = C + D
        # Apply PC-2 permutation to get 48-bit subkey
        round_key = permute(combined, PC2)
        round_keys.append(round_key)
    return round_keys


# The DES f-function (Feistel function)
def feistel(right, round_key):
    """Feistel function: expand, XOR with round key, substitute using S-boxes, and permute using P."""
    # Expansion from 32 bits to 48 bits
    expanded_right = permute(right, E)
    # XOR with round key
    temp = xor(expanded_right, round_key)
    
    # Substitution using 8 S-boxes. Each group is 6 bits.
    sub_output = []
    for i in range(8):
        # Take 6 bits for the current S-box
        block = temp[i * 6:(i + 1) * 6]
        # The first and last bits form the row (0-3)
        row = block[0] * 2 + block[5]
        # Middle 4 bits form the column (0-15)
        col = block[1] * 8 + block[2] * 4 + block[3] * 2 + block[4]
        # Get value from S_BOX; convert it into 4-bit binary list
        s_val = S_BOX[i][row][col]
        bits = [ (s_val >> j) & 1 for j in [3, 2, 1, 0] ]
        sub_output.extend(bits)
    
    # Final permutation using the P table.
    output = permute(sub_output, P)
    return output


# The main DES function: can be used for encryption (for decryption, simply reverse the round keys)
def des(data, key, decrypt=False):
    """
    data: a hexadecimal string of 16 hex digits (64 bits)
    key: a hexadecimal string of 16 hex digits (64 bits)
    decrypt: set True for decryption
    Returns a hexadecimal string (ciphertext or plaintext).
    """
    # Convert input hex to binary (list of 0s and 1s)
    data_bits = hex_to_bin(data)
    key_bits = hex_to_bin(key)

    # Generate 16 round keys
    round_keys = generate_keys(key_bits)
    if decrypt:
        round_keys = round_keys[::-1]  # reverse keys for decryption
    
    # Initial permutation on data block
    data_bits = permute(data_bits, IP)
    # Split data block into left (L) and right (R) 32-bit halves
    L = data_bits[:32]
    R = data_bits[32:]
    
    # 16 rounds of the Feistel structure:
    for round_key in round_keys:
        temp_R = R[:]  # copy current R
        # f-function on R with the round key
        f_out = feistel(R, round_key)
        # new R becomes L XOR f_out
        R = xor(L, f_out)
        # new L becomes old R
        L = temp_R
    
    # After 16 rounds, reverse the order (note: no swap in final step)
    combined = R + L
    # Final permutation to get the result
    final_data = permute(combined, FP)
    # Convert binary list back to hexadecimal string
    return bin_to_hex(final_data)


# Example usage:
if __name__ == '__main__':
    # 64-bit plaintext and key represented as 16-hex-digit strings
    plaintext = "0123456789ABCDEF"  # example plaintext
    key = "133457799BBCDFF1"        # example key
    
    print("Plaintext: ", plaintext)
    print("Key:       ", key)
    
    # Encrypt the plaintext
    ciphertext = des(plaintext, key)
    print("Ciphertext:", ciphertext)
    
    # Decrypt to recover the original plaintext
    recovered_plaintext = des(ciphertext, key, decrypt=True)
    print("Decrypted: ", recovered_plaintext)
