/// Base calculator class that performs basic arithmetic operations.
/// Used to test inheritance edges in the graph.
public class Calculator {
    
    public init() {}
    
    /// Performs addition of two numbers.
    public func calculate(_ a: Double, _ b: Double, operation: String) -> Double {
        switch operation {
        case "+": return add(a, b)
        case "-": return subtract(a, b)
        case "*": return multiply(a, b)
        case "/": return divide(a, b)
        default: return 0
        }
    }
    
    public func add(_ a: Double, _ b: Double) -> Double {
        return a + b
    }
    
    public func subtract(_ a: Double, _ b: Double) -> Double {
        return a - b
    }
    
    public func multiply(_ a: Double, _ b: Double) -> Double {
        return a * b
    }
    
    public func divide(_ a: Double, _ b: Double) -> Double {
        guard b != 0 else { return .nan }
        return a / b
    }
}
