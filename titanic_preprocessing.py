import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
titanic_df = pd.read_csv("titanic.csv")

# Drop irrelevant columns
titanic_df.drop(["who", "adult_male", "deck", "embark_town", "alive", "class"], axis=1, inplace=True)

# Handle missing values
# Impute 'age' with the mean
imputer_age = SimpleImputer(strategy="mean")
titanic_df["age"] = imputer_age.fit_transform(titanic_df[["age"]])

# Impute 'embarked' with the most frequent value
imputer_embarked = SimpleImputer(strategy="most_frequent")
titanic_df["embarked"] = imputer_embarked.fit_transform(titanic_df[["embarked"]]).ravel()

# Encode categorical features
# 'sex' column
le = LabelEncoder()
titanic_df["sex"] = le.fit_transform(titanic_df["sex"])

# 'embarked' column
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
embarked_encoded = ohe.fit_transform(titanic_df[["embarked"]])
embarked_df = pd.DataFrame(embarked_encoded, columns=ohe.get_feature_names_out(["embarked"]))
titanic_df = pd.concat([titanic_df, embarked_df], axis=1)
titanic_df.drop("embarked", axis=1, inplace=True)

# Feature Scaling for 'fare' and 'age'
scaler = StandardScaler()
titanic_df[["fare", "age"]] = scaler.fit_transform(titanic_df[["fare", "age"]])

# Define features (X) and target (y)
X = titanic_df.drop("survived", axis=1)
y = titanic_df["survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessed data head:")
print(X_train.head())
print("\nPreprocessed data info:")
X_train.info()

# Save preprocessed data for model training
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Save the imputer, encoder, and scaler for later use in prediction
import joblib
joblib.dump(imputer_age, "imputer_age.pkl")
joblib.dump(imputer_embarked, "imputer_embarked.pkl")
joblib.dump(le, "label_encoder_sex.pkl")
joblib.dump(ohe, "one_hot_encoder_embarked.pkl")
joblib.dump(scaler, "scaler.pkl")


