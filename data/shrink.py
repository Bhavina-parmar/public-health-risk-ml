import pandas as pd

print("📥 Reading large CSV...")
df = pd.read_csv("data/raw.csv", nrows=20000)
   # take first 20k only
print("✔ Loaded:", df.shape)

print("📤 Saving smaller dataset...")
df.to_csv("data/small.csv", index=False)


print("🎉 Done! Saved as small.csv")
