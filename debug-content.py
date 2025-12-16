import os
from pathlib import Path

# Read the original file
source_file = Path("book/01-introduction/embodied-intelligence/content.md")
with open(source_file, 'r', encoding='utf-8') as f:
    original_content = f.read()

print("First 100 characters of original file:")
print(repr(original_content[:100]))
print()
print("First 10 lines of original file:")
for i, line in enumerate(original_content.split('\n')[:10]):
    print(f"Line {i}: {repr(line)}")