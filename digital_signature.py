import hashlib
import random

#pow(a,-1,m) alternative
# Helper Functions
def modinv(a, m):
    # Modular inverse using extended Euclidean algorithm
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m 
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 % m0

def hash_message(message):
    return int(hashlib.sha1(message.encode()).hexdigest(), 16)

# Step 1: Global parameters (small primes for demo)
p = 30803  # Large prime number
q = 491    # Prime divisor of p-1
g = 2      # Generator (g^q mod p == 1)

# Step 2: Key generation
x = random.randint(1, q - 1)  # Private key
y = pow(g, x, p)              # Public key

print("Private key (x):", x)
print("Public key (y):", y)

# Step 3: Sign a message
def sign(message):
    Hm = hash_message(message)
    
    while True:
        k = random.randint(1, q - 1)#  Random nonce
        r = pow(g, k, p) % q
        if r == 0:
            continue
        k_inv = modinv(k, q)
        s = (k_inv * (Hm + x * r)) % q
        if s == 0:
            continue
        break
    return (r, s)

# Step 4: Verify a signature
def verify(message, r, s):
    if not (0 < r < q and 0 < s < q):
        return False
    
    Hm = hash_message(message)
    w = modinv(s, q)
    u1 = (Hm * w) % q
    u2 = (r * w) % q
    v = ((pow(g, u1, p) * pow(y, u2, p)) % p) % q
    return v == r

# Demo
message = "Hello, this is a signed message!"
r, s = sign(message)
print("\nSignature:")
print("r =", r)
print("s =", s)

valid = verify(message, r, s)
print("\nVerification:", " Valid Signature" if valid else " Invalid Signature")
