/// Engine that uses constructor-level dependency injection.
/// Tests DI pattern and complex CALLS edges.
public class Engine {
    
    private let calculator: Calculator
    private let factory: OperationFactory
    
    /// Constructor injection: takes Calculator and OperationFactory.
    public init(calculator: Calculator, factory: OperationFactory) {
        self.calculator = calculator
        self.factory = factory
    }
    
    /// Convenience init with default dependencies.
    public convenience init() {
        self.init(calculator: Calculator(), factory: OperationFactory())
    }
    
    /// Evaluates an expression using the injected dependencies.
    /// Tests call chain: evaluate -> factory.createOperation -> operation.perform
    public func evaluate(_ a: Double, _ b: Double, symbol: String) -> Double {
        if let operation = factory.createOperation(for: symbol) {
            return performWithOperation(a, b, operation: operation)
        }
        return calculator.calculate(a, b, operation: symbol)
    }
    
    private func performWithOperation(_ a: Double, _ b: Double, operation: Operation) -> Double {
        let result = operation.perform(a, b)
        return logResult(result)
    }
    
    private func logResult(_ result: Double) -> Double {
        // In a real app, this might log to analytics
        print("Result: \(result)")
        return result
    }
}
