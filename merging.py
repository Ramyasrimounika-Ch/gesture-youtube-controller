import os
import pandas as pd

folder_path = r"D:/youtube_ai/Dataa"
merged = []

files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
files.sort()

label_map = {file: i for i, file in enumerate(files)}

for file in files:
    path = os.path.join(folder_path, file)
    df = pd.read_csv(path)
    df["label"] = label_map[file]
    merged.append(df)

final_df = pd.concat(merged, ignore_index=True)
final_df.to_csv(r"D:/youtube_ai/final_dataset.csv", index=False)

for file, label in label_map.items():
    print(f"{label} → {os.path.splitext(file)[0]}")
