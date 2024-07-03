import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
filename = 'training_data/random_superheroes_supervillains_with_roles.csv'
data = pd.read_csv(filename)

# Preprocessing the data
# Encoding categorical features
le_passion = LabelEncoder()
data['Passion'] = le_passion.fit_transform(data['Passion'])

le_hair_color = LabelEncoder()
data['Hair Color'] = le_hair_color.fit_transform(data['Hair Color'])

le_powers = LabelEncoder()
data['Powers'] = le_powers.fit_transform(data['Powers'])

le_home = LabelEncoder()
data['Home'] = le_home.fit_transform(data['Home'])

le_type = LabelEncoder()
data['Role'] = le_type.fit_transform(data['Role'])

# Splitting features and target
X = data.drop(['Name', 'Role'], axis=1)
y = data['Role']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the classifier
classifier = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=50, min_samples_split=2, min_samples_leaf=1)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate a classification report
class_report = classification_report(y_test, y_pred)

# Display the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le_type.classes_, yticklabels=le_type.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Display feature importances
feature_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("Feature Importances:")
print(feature_importances)

# Output the classification report
print("Classification Report:")
print(class_report)

# Function to classify a new input
def classify_new_input(input_data):
    input_df = pd.DataFrame([input_data], columns=['Passion', 'Hair Color', 'Powers', 'Home'])
    input_df['Passion'] = le_passion.transform(input_df['Passion'])
    input_df['Hair Color'] = le_hair_color.transform(input_df['Hair Color'])
    input_df['Powers'] = le_powers.transform(input_df['Powers'])
    input_df['Home'] = le_home.transform(input_df['Home'])
    
    prediction = classifier.predict(input_df)
    predicted_type = le_type.inverse_transform(prediction)
    
    return predicted_type[0]

print("TESTING SUPERHEROES:")

filename = 'test_data/superheroes.csv'
data = pd.read_csv(filename)

new_inputs = data.to_dict(orient='records')

for new_input in new_inputs:
  predicted_class = classify_new_input(new_input)
  print(f"{predicted_class}: {new_input}")

print ("TESTING VILLAINS:")
filename = 'test_data/supervillains.csv'
data = pd.read_csv(filename)

new_inputs = data.to_dict(orient='records')

for new_input in new_inputs:
  predicted_class = classify_new_input(new_input)
  print(f"{predicted_class}: {new_input}")

additional_test = {
   "Passion": "High",
   "Hair Color": "Blond",
   "Powers": "Strength",
   "Home": "Lair"
}

predicted_class = classify_new_input(additional_test)

print(f"{predicted_class}: {additional_test}")