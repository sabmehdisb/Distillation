import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. Dataset Loading ---
# Replace "Recidivism" with the actual target variable name if needed.
dataset_path = 'balance-scale_0training_data.csv'
df = pd.read_csv(dataset_path)

# Assuming the target column is the last column and all other columns are features
X = df.drop(df.columns[-1], axis=1)  # remove the last column
y = df[df.columns[-1]]               # take values from the last column

# --- 2. Split into train / test sets (here we use only training) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --- 3. Random Forest Training ---
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
rf.fit(X_train, y_train)

# --- 4. Function to calculate node depths ---
def compute_node_depths(tree_):
    """Calculates the depth of each node in the decision tree (returns a list of integers)."""
    node_count = tree_.node_count
    depths = [-1] * node_count
    # Use a stack for depth-first traversal (DFS)
    stack = [(0, 0)]  # (node_id, depth), root node is 0
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left != -1:
            stack.append((left, depth + 1))
        if right != -1:
            stack.append((right, depth + 1))
    return depths

# --- 5. Convert tree to text format ---
def tree_to_text(rf_tree):
    """
    Returns a string describing the tree structure in the format:
      <node_index> <node type> <left child> <right child> <feature> <threshold> <node_depth> <majority class>
    For an internal node (IN), the "majority class" is displayed as -1.
    For a leaf node (LN), feature and threshold are -1 and the majority class is the predicted class.
    """
    tree_ = rf_tree.tree_
    depths = compute_node_depths(tree_)
    lines = []
    node_count = tree_.node_count
    for node in range(node_count):
        # If the node has no children, it's a leaf
        if tree_.children_left[node] == -1 and tree_.children_right[node] == -1:
            node_type = "LN"
            left_child = -1
            right_child = -1
            feature = -1
            threshold = -1
            # Majority is the class label predicted, determined by argmax of tree_.value[node]
            majority = int(np.argmax(tree_.value[node]))
        else:
            node_type = "IN"
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            feature = int(tree_.feature[node])
            threshold = float(tree_.threshold[node])
            majority = -1  # By convention, for an internal node, we put -1
        depth = depths[node]
        # Create the line in the requested format
        # Format: node_index <space> node_type <space> left_child <space> right_child <space> feature <space> threshold <space> depth <space> majority_class
        line = f"{node} {node_type} {left_child} {right_child} {feature} {threshold if threshold != -1 else -1} {depth} {majority}"
        lines.append(line)
    return "\n".join(lines)

# --- 6. Writing the output file ---
ensemble = 'RF'
nb_trees = rf.n_estimators
nb_features = X.shape[1]
nb_classes = len(np.unique(y))
max_tree_depth = rf.max_depth

output_file = dataset_path+".txt"
with open(output_file, "w") as f:
    f.write(f"DATASET_NAME: {dataset_path}\n")
    f.write(f"ENSEMBLE: {ensemble}\n")
    f.write(f"NB_TREES: {nb_trees}\n")
    f.write(f"NB_FEATURES: {nb_features}\n")
    f.write(f"NB_CLASSES: {nb_classes}\n")
    f.write(f"MAX_TREE_DEPTH: {max_tree_depth}\n")
    f.write("Format: node / node type (LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n")
    
    for tree_idx, estimator in enumerate(rf.estimators_):
        f.write(f"\n[TREE {tree_idx}]\n")
        f.write(f"NB_NODES: {estimator.tree_.node_count}\n")
        f.write(tree_to_text(estimator))
        f.write("\n")
        
print(f"File '{output_file}' created successfully.")