import pandas as pd
from sklearn.model_selection import train_test_split

#Read the dataset
df = pd.read_excel("data/Dry_Bean_Dataset.xlsx") 
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

test_df = X_test.copy()
test_df['Class'] = y_test.values

test_df.to_csv("data/test.csv", index=False)