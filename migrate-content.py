import os
import shutil
import re
from pathlib import Path

def add_frontmatter(content, title, sidebar_position):
    """Add Docusaurus frontmatter to markdown content"""
    # Check if content already has frontmatter by looking for the pattern:
    # --- at start of document followed by metadata and ---
    lines = content.split('\n')
    has_frontmatter = False

    if len(lines) > 0 and lines[0].strip() == '---':
        # Look for the closing ---
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                # Check if between the --- markers there are key-value pairs (indication of frontmatter)
                potential_metadata = lines[1:i]
                if any(':' in line for line in potential_metadata):  # Check if there are key-value pairs
                    has_frontmatter = True
                break

    if has_frontmatter:
        # If content already has frontmatter, just return it as is
        return content
    else:
        # Check if title contains characters that need escaping in YAML
        # The colon is particularly problematic when followed by spaces
        if ':' in title or re.search(r'[\[\]{}],&*#?|<>=!%@]', title):
            # Quote the title to avoid YAML parsing issues
            title = f'"{title}"'

        # Add new frontmatter
        frontmatter = f"""---
title: {title}
sidebar_position: {sidebar_position}
---

"""
        # If content starts with --- (horizontal rule) followed by # heading, we need to handle this specially
        if content.startswith('---\n#'):
            # Remove the leading --- (horizontal rule) since we'll be adding proper frontmatter
            content = content[4:]  # Remove first 4 characters: "---\n"

        return frontmatter + content

def get_title_from_filename(filepath):
    """Generate a title from the filename/path"""
    filename = Path(filepath).stem
    if filename == "content":
        # If the filename is 'content', derive title from the parent directory
        parent_dir = Path(filepath).parent.name
        title = parent_dir.replace('-', ' ').title()
    else:
        title = filename.replace('-', ' ').title()
    return title

def migrate_book_content():
    """Migrate book content to Docusaurus structure"""
    source_dir = Path("book")
    dest_dir = Path("website/docs")
    
    # Define chapter mapping with proper ordering and titles
    chapter_map = {
        "01-introduction": "1. Introduction",
        "02-ros-middleware": "2. ROS 2 Middleware",
        "03-simulation": "3. Simulation Environments", 
        "04-vla-systems": "4. Vision-Language-Action Systems",
        "05-llm-planning": "5. LLM-Driven Planning",
        "06-humanoid-locomotion": "6. Humanoid Locomotion",
        "07-cognitive-robotics": "7. Cognitive Robotics",
        "08-integration": "8. System Integration",
        "09-capstone": "9. Capstone Project"
    }
    
    # Track sidebar positions
    sidebar_positions = {}
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.md'):
                source_file = Path(root) / file
                relative_path = source_file.relative_to(source_dir)
                
                # Determine destination path
                dest_file = dest_dir / relative_path
                
                # Create destination directory if it doesn't exist
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Read the original content
                with open(source_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # Extract the first heading to use as title if available
                title = get_title_from_filename(source_file)
                for line in original_content.split('\n'):
                    if line.startswith('# '):
                        title = line[2:]  # Remove '# ' prefix
                        break
                
                # Determine sidebar position based on the file structure
                path_parts = str(relative_path).split(os.sep)
                
                # Generate unique key for tracking sidebar positions
                if path_parts[0] in chapter_map:
                    chapter_key = path_parts[0]
                    section_key = f"{chapter_key}/{path_parts[1] if len(path_parts) > 1 else 'intro'}"
                    
                    if chapter_key not in sidebar_positions:
                        sidebar_positions[chapter_key] = 1
                    if section_key not in sidebar_positions:
                        sidebar_positions[section_key] = 1
                        
                    # Determine sidebar position
                    if 'exercises' in path_parts or 'setup' in path_parts:
                        # Exercises and setup files get higher numbers
                        position = sidebar_positions[section_key] + 100
                        sidebar_positions[section_key] += 1
                    else:
                        position = sidebar_positions[section_key]
                        sidebar_positions[section_key] += 1
                
                    # Add frontmatter to content
                    content_with_frontmatter = add_frontmatter(original_content, title, position)
                    
                    # Write to destination
                    with open(dest_file, 'w', encoding='utf-8') as f:
                        f.write(content_with_frontmatter)
                    
                    print(f"Migrated: {source_file} -> {dest_file}")

if __name__ == "__main__":
    migrate_book_content()
    print("Content migration completed!")