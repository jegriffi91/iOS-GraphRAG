import Foundation

/// Phase 5 P2 fixture: exercises Swift subscripts and typealiases.
/// Asserts in `tests/test_phase5p2_coverage.py` pin the extracted symbols.

public typealias OperationFn = (Double, Double) -> Double
public typealias DigitMap = [Character: Int]

public class HistoryStorage {
    private var entries: [String: Double] = [:]

    public subscript(key: String) -> Double? {
        get { return entries[key] }
        set { entries[key] = newValue }
    }

    public subscript(index: Int, withDefault fallback: Double) -> Double {
        return Double(index) + fallback
    }
}

public struct Coordinate {
    public typealias Pair = (x: Int, y: Int)
    public let p: Pair
}
