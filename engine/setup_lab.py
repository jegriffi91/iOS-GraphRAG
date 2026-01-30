import os
from pathlib import Path

ROOT = Path("mock_repo")

def create_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content.strip())

def setup_lab():
    print(f"🧪 Generating Synthetic iOS Repo at {ROOT}...")
    
    # 1. The Core Definition (ObjC)
    create_file(ROOT / "Core/AuthSession.h", """
    @interface AuthSession : NSObject
    @property (nonatomic, copy) NSString *token;
    - (BOOL)isValid;
    @end
    """)

    # 2. The Swift Consumer
    create_file(ROOT / "Features/Cart/CartManager.swift", """
    import Foundation
    import Core

    class CartManager {
        let session: AuthSession
        
        func checkout() {
            guard session.isValid() else { return }
            print("Checking out...")
        }
    }
    """)

    # 3. The Complexity: Swift Extension of ObjC Class in a different module
    create_file(ROOT / "Features/Auth/AuthSession+Premium.swift", """
    import Core

    extension AuthSession {
        /// Checks if the user has a premium token
        var isPremium: Bool {
            return self.token.hasPrefix("PREM")
        }
    }
    """)

    print("✅ Lab Ready. Challenge: Link 'AuthSession+Premium.swift' to 'AuthSession.h'.")

if __name__ == "__main__":
    setup_lab()
