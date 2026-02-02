/// State machine that demonstrates function call chains.
/// Tests CALLS edges: start -> process -> complete
public class StateMachine {
    
    public enum State {
        case idle
        case processing
        case completed
        case failed
    }
    
    private var state: State = .idle
    private var value: Double = 0
    
    public init() {}
    
    /// Entry point: starts the state machine.
    /// Call chain: start -> validate -> process
    public func start(with input: Double) -> State {
        state = .idle
        value = input
        
        if validate(input) {
            return process()
        } else {
            return handleError("Invalid input")
        }
    }
    
    /// Validates the input before processing.
    private func validate(_ input: Double) -> Bool {
        return !input.isNaN && !input.isInfinite
    }
    
    /// Processes the input and transitions state.
    /// Call chain: process -> transform -> complete
    private func process() -> State {
        state = .processing
        value = transform(value)
        return complete()
    }
    
    /// Transforms the value (example: doubles it).
    private func transform(_ input: Double) -> Double {
        return input * 2
    }
    
    /// Completes the state machine.
    private func complete() -> State {
        state = .completed
        notifyObservers()
        return state
    }
    
    /// Handles errors and transitions to failed state.
    private func handleError(_ message: String) -> State {
        state = .failed
        logError(message)
        return state
    }
    
    private func notifyObservers() {
        // Notify any observers of completion
        print("State machine completed with value: \(value)")
    }
    
    private func logError(_ message: String) {
        print("Error: \(message)")
    }
    
    /// Returns the current state.
    public func currentState() -> State {
        return state
    }
    
    /// Returns the current value.
    public func currentValue() -> Double {
        return value
    }
}
