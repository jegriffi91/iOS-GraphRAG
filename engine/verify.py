import time
from indexer import build_graph
from server import hydrate, read_live_code, NODE_IDS

def verify():
    print("1️⃣  Building Index...")
    build_graph()
    
    print("2️⃣  Starting Server...")
    hydrate()
    
    # Find the Swift Extension node
    # ID format: path::Symbol
    ext_node = [n for n in NODE_IDS if "AuthSession+Premium" in n][0]
    print(f"Targeting: {ext_node}")
    
    # Test 1: Read Live
    print("3️⃣  Verifying Disk Read...")
    code = read_live_code(ext_node)
    assert "isPremium" in code
    print("✅ Live Read Successful.")
    
    # Test 2: Modify Disk (Simulate Developer Edit)
    print("4️⃣  Simulating Edit...")
    p = "mock_repo/Features/Auth/AuthSession+Premium.swift"
    with open(p, "r") as f: content = f.read()
    
    # Mutate the file
    with open(p, "w") as f: f.write(content.replace("isPremium", "isSuperPremium"))
    
    # Test 3: Read Again (Without Re-indexing)
    new_code = read_live_code(ext_node)
    if "isSuperPremium" in new_code:
        print("✅ SUCCESS: Server reflected live edit immediately.")
    else:
        print("❌ FAILED: Server served stale content.")

if __name__ == "__main__":
    verify()
