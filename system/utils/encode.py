# LFSR pseudo-random number generator
class LFSR:
    def __init__(self, seed, taps):
        self.register = seed
        self.taps = taps

    def step(self):
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.register >> tap) & 1

        self.register = (self.register << 1) | feedback
        self.register &= (1 << len(self.taps)) - 1

        return feedback  # Return feedback bit as pseudo-random bit

    def generate_keystream(self, count):
        return [self.step() for _ in range(count)]


def xor_encrypt(plaintext, keystream):
    return [p ^ k for p, k in zip(plaintext, keystream)]


# Example usage
if __name__ == "__main__":
    seed = 0b101001  # 6-bit seed
    taps = [5, 3]  # Feedback bit positions
    lfsr = LFSR(seed, taps)

    random_numbers = lfsr.generate_random_numbers(10)
    print(random_numbers)
