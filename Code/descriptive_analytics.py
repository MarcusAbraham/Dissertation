import pandas as pd


def count_side_effects(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Count the number of unique values in the specified column
    unique_side_effects = df["Side Effect Name"].nunique()

    print(f"Unique side effects: {unique_side_effects}")


def count_unique_values(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Count the number of unique values in the specified column
    unique_values_count = df["Side Effect Name"].value_counts()

    # Display unique values and their counts
    print("Unique Values and Their Counts:")
    for id, (value, count) in enumerate(unique_values_count.items(), start=0):
        print(f"{value} (Count: {count})")


def filter_data(filename):
    # Read CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Count unique values in the side effect column
    counts = df.iloc[:, 13].value_counts()

    # Filter rows based on counts
    #filtered_df = df[df.iloc[:, 13].isin(counts[counts >= 50].index)]
    filtered_df = df[df.iloc[:, 13].isin(counts[counts >= 4200].index)]

    # Write filtered data to a new CSV file
    filtered_df.to_csv("ChChSe-Decagon_polypharmacy/filteredData.csv", index=False)


# Call the functions to count unique values
#unique_effects = count_side_effects("ChChSe-Decagon_polypharmacy/computedData.csv")
#count_unique_values("ChChSe-Decagon_polypharmacy/computedData.csv")

#Filter the data to remove any side effects with less than 50 pairs
#filter_data("ChChSe-Decagon_polypharmacy/computedData.csv")
