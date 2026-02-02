/// Factory for creating Operation instances.
/// Tests the factory pattern and CALLS edges.
public class OperationFactory {
    
    public init() {}
    
    /// Creates an operation based on the symbol.
    /// Tests function call: createOperation -> specific init calls
    public func createOperation(for symbol: String) -> Operation? {
        switch symbol {
        case "+": return createAddition()
        case "-": return createSubtraction()
        case "*": return createMultiplication()
        case "/": return createDivision()
        default: return nil
        }
    }
    
    private func createAddition() -> Addition {
        return Addition()
    }
    
    private func createSubtraction() -> Subtraction {
        return Subtraction()
    }
    
    private func createMultiplication() -> Multiplication {
        return Multiplication()
    }
    
    private func createDivision() -> Division {
        return Division()
    }
}
