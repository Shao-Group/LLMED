from ete3 import Tree
import sys

def calculate_rf(tree_file_1, tree_file_2):
    # Load the trees

    t1 = Tree(tree_file_1)
    t2 = Tree(tree_file_2)
    rf_results = t1.robinson_foulds(t2, unrooted_trees=True)
    
    rf = rf_results[0]
    max_rf = rf_results[1]
    common_attrs = rf_results[2] # Shared bipartitions
    
    n_rf = rf / max_rf if max_rf > 0 else 0

    print(f"Raw RF Score: {rf}")
    print(f"Max possible RF: {max_rf}")
    print(f"Normalized RF Score: {n_rf:.4f}")
    print(f"Number of shared splits: {len(common_attrs)}")

# Usage
calculate_rf(sys.argv[1], sys.argv[2])