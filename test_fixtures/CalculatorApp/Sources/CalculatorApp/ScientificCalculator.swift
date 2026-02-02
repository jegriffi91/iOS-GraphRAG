import Foundation

/// Scientific calculator that extends the base Calculator.
/// Tests INHERITS edge: ScientificCalculator -> Calculator
public class ScientificCalculator: Calculator {
    
    public override init() {
        super.init()
    }
    
    /// Override to add scientific operations.
    public override func calculate(_ a: Double, _ b: Double, operation: String) -> Double {
        switch operation {
        case "^": return power(a, b)
        case "sqrt": return squareRoot(a)
        case "log": return logarithm(a)
        default: return super.calculate(a, b, operation: operation)
        }
    }
    
    public func power(_ base: Double, _ exponent: Double) -> Double {
        return pow(base, exponent)
    }
    
    public func squareRoot(_ value: Double) -> Double {
        return sqrt(value)
    }
    
    public func logarithm(_ value: Double) -> Double {
        return log(value)
    }
    
    /// Calculates factorial using recursion.
    /// Tests function call chain: factorial -> factorial (recursive)
    public func factorial(_ n: Int) -> Int {
        guard n > 1 else { return 1 }
        return n * factorial(n - 1)
    }
}
