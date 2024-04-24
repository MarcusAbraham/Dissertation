import pandas as pd


# Function to count the number of unique side effects in the dataset
def count_side_effects(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Count the number of unique values in the side effect column
    unique_side_effects = df["Side Effect Name"].nunique()

    print(f"Unique side effects: {unique_side_effects}")


# Function to count the amount of drug pairs for each side effect
def count_unique_values(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Count the number of samples for each unique value in the side effect column
    unique_values_count = df["Side Effect Name"].value_counts()

    # Display unique values and their counts
    print("Unique Values and Their Counts:")
    for id, (value, count) in enumerate(unique_values_count.items(), start=0):
        print(f"{value} (Count: {count})")


# Function to filter the side effects with few samples
def filter_data(filename):
    # Read CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Count unique values in the side effect column
    counts = df.iloc[:, 13].value_counts()

    # Filter for the top 3 side effects
    filtered_df = df[df.iloc[:, 13].isin(counts[counts >= 4200].index)]

    # Write filtered data to a new CSV file
    filtered_df.to_csv("ChChSe-Decagon_polypharmacy/filteredData.csv", index=False)


# Call the functions to count unique values
count_side_effects("ChChSe-Decagon_polypharmacy/computedData.csv")
count_unique_values("ChChSe-Decagon_polypharmacy/computedData.csv")

#Filter the data
filter_data("ChChSe-Decagon_polypharmacy/computedData.csv")

