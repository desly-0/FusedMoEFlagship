"""Generate C++ file with embedded kernel source as string literal."""
import sys


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <kernel_src> <tiling_header> <output_cpp>")
        sys.exit(1)

    kernel_path = sys.argv[1]
    tiling_path = sys.argv[2]
    out_path = sys.argv[3]

    with open(kernel_path, "r") as f:
        kernel_src = f.read()

    with open(tiling_path, "r") as f:
        tiling_src = f.read()

    # Generate C++ source with embedded kernel source string
    # Uses RTC (runtime compilation) per CANN dev_guide §2.3.1.5
    with open(out_path, "w") as f:
        f.write("// Auto-generated kernel source string for RTC\n")
        f.write('#include <string>\n\n')
        f.write('// Concatenated kernel source (tiling header inlined)\n')
        f.write('// aclrtcCompileProg uses bisheng internally, finds CANN headers\n')
        f.write('const char* g_kernelSource = R"ascendc(\n')

        # Inline tiling header (replace #include "fused_moe_tiling.h")
        for line in tiling_src.splitlines(True):
            if '#pragma once' in line:
                continue
            f.write(line)

        # Write kernel source, skip #include of tiling header
        f.write('\n')
        for line in kernel_src.splitlines(True):
            if '#include "fused_moe_tiling.h"' in line:
                continue
            f.write(line)

        f.write('\n)ascendc";\n')

    chars = len(kernel_src) + len(tiling_src)
    print(f"Embedded kernel source ({chars} chars) -> {out_path}")


if __name__ == "__main__":
    main()
