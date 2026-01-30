import tree_sitter_swift
try:
    print(f"Version: {tree_sitter_swift.__version__}")
except:
    print("No __version__")
print(f"Language: {tree_sitter_swift.language()}")
