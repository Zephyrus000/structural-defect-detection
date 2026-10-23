
from pathlib import Path
from collections import Counter

labels_dir = Path("data/CUBIT-InSeg/train/labels")
counts = Counter()

for file in labels_dir.rglob("*.txt"):
    for line in file.read_text().splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            counts[class_id] += 1

total = sum(counts.values())

for class_id, count in sorted(counts.items()):
    print(f"Class {class_id}: {count} ({count/total:.1%})")
