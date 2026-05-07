import SwiftUI

/// Phase 5 P1 fixture: exercises enum cases (bare, raw-valued, associated
/// values) plus a SwiftUI-style modifier chain with a trailing closure
/// containing inner calls. Asserts in `tests/test_phase5p1_coverage.py`
/// pin the extracted symbols and CALLS edges.

public enum CalculatorOperation {
    case add
    case subtract
    case multiply
    case divide
    case custom(name: String, fn: (Double, Double) -> Double)
    case configured(power: Int = 2)
}

public enum LoadingState: String {
    case idle = "idle"
    case loading = "loading"
    case error = "error"
}

public struct ChainExample {
    public func process() {
        Text("Hello")
            .padding()
            .background(Color.red)
            .onTapGesture {
                handleTap()
                state.update()
            }
    }

    private func handleTap() {}
    private var state: AnyObject { return self }
}
