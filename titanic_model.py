import seaborn as sns
import pandas as pd

# Load the Titanic dataset
titanic_df = sns.load_dataset("titanic")

# Display the first few rows of the dataset
print("Head of the Titanic dataset:")
print(titanic_df.head())

# Display information about the dataset, including data types and non-null values
print("\nInfo of the Titanic dataset:")
titanic_df.info()

# Save the dataframe to a CSV file for later use
titanic_df.to_csv("titanic.csv", index=False)


