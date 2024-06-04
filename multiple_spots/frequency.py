import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def count_strains(folder_path):
    strain_count = {}
    total_files = 0

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            total_files += 1
            # Extract strain name from filename and remove '.csv'
            parts = filename.split("_")
            if len(parts) > 3:
                strain_name = parts[3].replace('.csv', '')
                # Increment count for strain
                strain_count[strain_name] = strain_count.get(strain_name, 0) + 1
            else:
                print(f"Filename {filename} does not have expected format")

    return strain_count, total_files

def calculate_strain_percentages(d1, d2):
    fractions = {x: float(d2.get(x, 0)) / d1[x] for x in d1}
    fractions.update((x, y * 100) for x, y in fractions.items())
    return fractions

def plot_strain_percentages(directories, labels):
    all_strain_percentages = []

    for direc, label in zip(directories, labels):
        folder_1_spot = os.path.join(direc, "1_spot/")
        folder_2_spots = os.path.join(direc, "2_spots/")

        # Count strains in each folder
        strain_count_1_spot, total_files_1_spot = count_strains(folder_1_spot)
        strain_count_2_spots, total_files_2_spots = count_strains(folder_2_spots)

        # Calculate percentages for each strain
        strain_percentages_combined = calculate_strain_percentages(strain_count_1_spot, strain_count_2_spots)

        # Create a DataFrame for the combined results
        df_combined = pd.DataFrame({
            "Strain": list(strain_percentages_combined.keys()),
            "Percentage": list(strain_percentages_combined.values()),
            "Label": label
        })

        # Append to the list
        all_strain_percentages.append(df_combined)

    # Concatenate all DataFrames
    df = pd.concat(all_strain_percentages, ignore_index=True)

    return df


# Example usage
# Example usage
direc = ''
directories = [direc + 'properties_T=3.5,Tmin=2.8/']
labels = [""]  # Labels for directories

df = plot_strain_percentages(directories, labels)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Strain", y="Percentage", hue="Label", palette="Set3", alpha=0.8)

# Add labels and title
plt.title("Percentage of events with multiple spots")
plt.xlabel("Strain")
plt.ylabel("Percentage of cells having multiple spots")
plt.xticks(rotation=45, ha='right')

# Show plot
plt.ylim([0,100])
plt.tight_layout()
plt.show()
