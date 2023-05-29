import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("/content/term-deposit-marketing-2020.csv")
df = df.drop(["job", "contact", "day", "month"], axis=1)

# Get the unique items used in one-hot encoding columns
marital = df["marital"].unique()
education = df["education"].unique()

encoder = OneHotEncoder()

# Perform one-hot encoding on "marital" and "education" columns
encoder_marital = pd.DataFrame(encoder.fit_transform(df[["marital"]]).toarray(), columns = marital)
encoder_edu = pd.DataFrame(encoder.fit_transform(df[["education"]]).toarray(), columns = education)

# Merge one-hot encoded columns back with original DataFrame
df = df.join(encoder_marital)
df = df.join(encoder_edu)
df.rename(columns={'unknown':'unknown_edu'}, inplace=True)
df = df.drop(["marital", "education"], axis=1)

df = df.replace(to_replace=['no', 'yes'], value=[0, 1])

y = df[["y"]]
X = df.drop(["y"], axis=1)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Get outliers
threshold = 3
z_scores = stats.zscore(X_smote[["age", "balance", "duration"]])
outlier_indices = (z_scores > threshold).any(axis=1)
outliers = X_smote[outlier_indices]
X_smote = X_smote[~outlier_indices]
y_smote = y_smote[~outlier_indices]

# Feature scaling
scaler = StandardScaler()
X_smote.loc[:, ["age", "balance", "duration"]] = scaler.fit_transform(X_smote[["age", "balance", "duration"]])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
