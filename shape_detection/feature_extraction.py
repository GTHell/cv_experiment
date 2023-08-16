import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Assuming you have a dataset of convex hull points and corresponding shape labels
convex_hull_points = [...]  # Convex hull points for each sample
shape_labels = [...]  # Shape labels for each sample

# Step 1: Convert shape labels to numeric form
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(shape_labels)

# Step 2: Feature Extraction and Normalization
features = []
for points in convex_hull_points:
  # Perform feature extraction on each set of convex hull points
  # Example feature extraction techniques:
  
  # Calculate the angles between consecutive points
  angles = []
  for i in range(len(points)):
      p1 = points[i]
      p2 = points[(i + 1) % len(points)]
      angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
      angles.append(angle)
  
  # Compute the distances between points
  distances = []
  for i in range(len(points)):
      p1 = points[i]
      p2 = points[(i + 1) % len(points)]
      distance = np.linalg.norm(p2 - p1)
      distances.append(distance)
  
  # Calculate the area of the convex hull
  area = 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))

  # Combine the extracted features into a single feature vector
  extracted_features = np.hstack((angles, distances, area))
  features.append(extracted_features)

# Convert the features and labels to numpy arrays
features = np.array(features)
numeric_labels = np.array(numeric_labels)

# Step 3: Feature Normalization
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
