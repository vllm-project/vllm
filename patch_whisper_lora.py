#!/usr/bin/env python3
"""
Patch installed vllm to add multi-LoRA support for Whisper models
"""

import os
import re
import sys


def find_vllm_whisper_file():
    """Find the installed vllm whisper.py file"""
    try:
        import vllm
        vllm_path = os.path.dirname(vllm.__file__)
        whisper_file = os.path.join(vllm_path, 'model_executor', 'models', 'whisper.py')

        if not os.path.exists(whisper_file):
            print(f"Error: whisper.py not found at {whisper_file}")
            return None

        return whisper_file
    except ImportError:
        print("Error: vllm not installed")
        return None


def backup_file(filepath):
    """Create a backup of the original file"""
    backup_path = filepath + '.backup'
    if not os.path.exists(backup_path):
        with open(filepath, 'r') as f:
            content = f.read()
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✓ Backup created: {backup_path}")
    else:
        print(f"✓ Backup already exists: {backup_path}")


def patch_whisper_file(filepath):
    """Patch the whisper.py file to add LoRA support"""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # 1. Add SupportsLoRA to imports
    import_pattern = r'from \.interfaces import (.*?)MultiModalEmbeddings(.*?)SupportsMultiModal(.*?)SupportsTranscription'

    if 'SupportsLoRA' not in content:
        # Find the import line and add SupportsLoRA
        content = re.sub(
            r'from \.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription',
            'from .interfaces import (\n    MultiModalEmbeddings,\n    SupportsLoRA,\n    SupportsMultiModal,\n    SupportsTranscription,\n)',
            content
        )
        print("✓ Added SupportsLoRA import")
    else:
        print("✓ SupportsLoRA import already present")

    # 2. Add SupportsLoRA to class definition
    class_pattern = r'class WhisperForConditionalGeneration\(\s*nn\.Module,\s*SupportsTranscription,\s*SupportsMultiModal\s*\):'

    if re.search(class_pattern, content):
        content = re.sub(
            class_pattern,
            'class WhisperForConditionalGeneration(\n    nn.Module, SupportsLoRA, SupportsTranscription, SupportsMultiModal\n):',
            content
        )
        print("✓ Added SupportsLoRA to class definition")
    elif 'class WhisperForConditionalGeneration(\n    nn.Module, SupportsLoRA' in content:
        print("✓ Class definition already patched")
    else:
        print("✗ Could not find class definition to patch")
        return False

    # 3. Add LoRA attributes after merge_by_field_config
    lora_attributes = '''    merge_by_field_config = True

    # LoRA support attributes
    supports_lora = True
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }
    embedding_modules = {
        "model.decoder.embed_tokens": "input_embeddings",
        "proj_out": "output_embeddings",
    }
    embedding_padding_modules = ["proj_out"]'''

    if 'supports_lora = True' not in content:
        # Find merge_by_field_config and add LoRA attributes after it
        content = re.sub(
            r'(class WhisperForConditionalGeneration\([^)]+\):\s+)merge_by_field_config = True\s+(packed_modules_mapping = \{)',
            r'\1' + lora_attributes + '\n\n    ',
            content,
            flags=re.MULTILINE
        )
        print("✓ Added LoRA attributes")
    else:
        print("✓ LoRA attributes already present")

    # Write the patched content
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"\n✅ Successfully patched {filepath}")
        return True
    else:
        print("\n✓ File already patched, no changes needed")
        return True


def verify_patch():
    """Verify that the patch worked"""
    try:
        from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

        checks = [
            ('supports_lora', hasattr(WhisperForConditionalGeneration, 'supports_lora') and
             WhisperForConditionalGeneration.supports_lora),
            ('embedding_modules', hasattr(WhisperForConditionalGeneration, 'embedding_modules')),
            ('embedding_padding_modules', hasattr(WhisperForConditionalGeneration, 'embedding_padding_modules')),
        ]

        print("\n" + "="*50)
        print("Verification:")
        all_passed = True
        for name, result in checks:
            status = "✓" if result else "✗"
            print(f"{status} {name}: {result}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n🎉 Whisper multi-LoRA support is ready!")
            return True
        else:
            print("\n❌ Some checks failed")
            return False

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False


def main():
    print("="*50)
    print("Whisper multi-LoRA Patch Script")
    print("="*50)

    # Find whisper.py file
    whisper_file = find_vllm_whisper_file()
    if not whisper_file:
        sys.exit(1)

    print(f"\nFound whisper.py at: {whisper_file}")

    # Backup original file
    backup_file(whisper_file)

    # Patch the file
    if not patch_whisper_file(whisper_file):
        print("\n❌ Patching failed")
        sys.exit(1)

    # Verify the patch
    if not verify_patch():
        sys.exit(1)

    print("\n" + "="*50)
    print("Done! You can now use Whisper with multi-LoRA:")
    print("vllm serve openai/whisper-v3-turbo \\")
    print("    --enable-lora \\")
    print("    --lora-modules lora1=/path/to/lora1 lora2=/path/to/lora2 \\")
    print("    --max-lora-rank 256")
    print("="*50)


if __name__ == '__main__':
    main()
