"""Convert binary file to C++ source with embedded byte array."""
import sys

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_binary> <output_cpp> [var_prefix]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "g_kernel"

    with open(in_path, "rb") as f:
        data = f.read()

    with open(out_path, "w") as f:
        f.write("// Auto-generated kernel binary blob\n")
        f.write('#include <cstddef>\n')
        f.write('#include <cstdint>\n')
        f.write('extern "C" {\n')
        f.write(f'const unsigned char {prefix}Binary[] = {{\n')
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"  {hex_bytes},\n")
        f.write('};\n')
        f.write(f'const size_t {prefix}BinarySize = sizeof({prefix}Binary);\n')
        f.write('}\n')

    print(f"Embedded {len(data)} bytes from {in_path} -> {out_path}")

if __name__ == "__main__":
    main()
