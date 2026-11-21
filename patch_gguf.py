"""Build-time compatibility fix for Krea2 GGUF standalone modulation parameters.

Pinned loader routes Linear weights through dequantization, but Krea2 uses
last.modulation.lin directly as a Parameter. Some converters quantize this
(2,6144) tensor to Q4_K storage (2,3456). Expand only this small parameter.
"""
from pathlib import Path

MARKER = '# FRAME Krea2 standalone modulation compatibility'
ANCHOR = '        # keep track of loaded tensor types\n'
PATCH = '''        # FRAME Krea2 standalone modulation compatibility
        # This parameter is used by SimpleModulation directly, not a GGUF Linear.
        if arch_str == "krea2" and sd_key.endswith("last.modulation.lin") and is_quantized(state_dict[sd_key]):
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

'''


def patch_source(source):
    if MARKER in source:
        return source
    if source.count(ANCHOR) != 1:
        raise RuntimeError('Pinned GGUF loader changed: compatibility patch requires review')
    return source.replace(ANCHOR, PATCH + ANCHOR, 1)


if __name__ == '__main__':
    path = Path('/opt/ComfyUI/custom_nodes/ComfyUI-GGUF/loader.py')
    path.write_text(patch_source(path.read_text()))
    print('KREA2_GGUF_MODULATION_PATCH_OK')
