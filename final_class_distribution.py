import pandas as pd

# ---------------------------------------------
# 1. Load the metadata CSV
# ---------------------------------------------
df = pd.read_csv("bovine_breeds_metadata.csv")

# Ensure breed names are clean
df["breed"] = df["breed"].str.strip()

# ---------------------------------------------
# 2. Define the final 16 cattle breeds used
# ---------------------------------------------
final_breeds = [
    "Ayrshire",
    "Banni",
    "Bargur",
    "Deoni",
    "Gir",
    "Guernsey",
    "Hallikar",
    "Holstein_Friesian",
    "Jersey",
    "Ongole",
    "Rathi",
    "Sahiwal",
    "Tharparkar",
    "Toda",
    "Umblachery",
    "Brown_Swiss"
]

# ---------------------------------------------
# 3. Filter rows belonging to only these breeds
# ---------------------------------------------
filtered_df = df[df["breed"].isin(final_breeds)]

# ---------------------------------------------
# 4. Count images per breed
# ---------------------------------------------
class_counts = (
    filtered_df["breed"]
    .value_counts()
    .reindex(final_breeds)   # keep order same
    .fillna(0)
    .astype(int)
)

# Convert to a table
output_df = class_counts.reset_index()
output_df.columns = ["Breed", "Image_Count"]

# ---------------------------------------------
# 5. Save to CSV
# ---------------------------------------------
output_df.to_csv("final_class_distribution.csv", index=False)

print("CSV file saved as: final_class_distribution.csv")
print(output_df)
