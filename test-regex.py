import re

# Test the regex
title = "Exercises: Introduction to Physical AI & Humanoid Robotics"
print(f"Testing title: {title}")
print(f"Characters that need escaping: {re.search(r'[:\[\]{}],&*#?|<>=!%@]', title)}")

if re.search(r'[:\[\]{}],&*#?|<>=!%@]', title):
    print("Title needs escaping")
    title = f'"{title}"'
    print(f"Escaped title: {title}")
else:
    print("Title does not need escaping")