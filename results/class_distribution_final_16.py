import pandas as pd

# Load metadata
df = pd.read_csv("bovine_breeds_metadata.csv")

# Your final 16 cattle breeds
final_breeds = [
    "Ayrshire", "Banni", "Bargur", "Brown", "Deoni",
    "Gir", "Guernsey", "Hallikar", "Holstein", "Jersey",
    "Ongole", "Rathi", "Sahiwal", "Tharparkar",
    "Toda", "Umblachery"
]

# Filter dataset
filtered_df = df[df['breed'].isin(final_breeds)]

# Count images
class_counts = filtered_df['breed'].value_counts().sort_index()

# Save table
class_counts.to_csv("class_distribution_final_16.csv")

# Display result
print(class_counts)
print("\nTotal images:", class_counts.sum())
