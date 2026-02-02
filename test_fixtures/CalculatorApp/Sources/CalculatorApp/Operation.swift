/// Protocol defining an arithmetic operation.
/// Tests CONFORMS edges for protocol conformance.
public protocol Operation {
    var symbol: String { get }
    func perform(_ a: Double, _ b: Double) -> Double
}

/// Addition operation conforming to Operation protocol.
public struct Addition: Operation {
    public let symbol = "+"
    
    public init() {}
    
    public func perform(_ a: Double, _ b: Double) -> Double {
        return a + b
    }
}

/// Subtraction operation conforming to Operation protocol.
public struct Subtraction: Operation {
    public let symbol = "-"
    
    public init() {}
    
    public func perform(_ a: Double, _ b: Double) -> Double {
        return a - b
    }
}

/// Multiplication operation conforming to Operation protocol.
public struct Multiplication: Operation {
    public let symbol = "*"
    
    public init() {}
    
    public func perform(_ a: Double, _ b: Double) -> Double {
        return a * b
    }
}

/// Division operation conforming to Operation protocol.
public struct Division: Operation {
    public let symbol = "/"
    
    public init() {}
    
    public func perform(_ a: Double, _ b: Double) -> Double {
        guard b != 0 else { return .nan }
        return a / b
    }
}
