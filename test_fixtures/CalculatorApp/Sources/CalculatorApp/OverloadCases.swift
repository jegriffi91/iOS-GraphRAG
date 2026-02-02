/// Test file for function overloading scenarios.
/// These three functions have the same base name but different selectors:
/// - test(_:)
/// - test(something:)
/// - test(somethingElse:)
public class OverloadTests {
    
    public init() {}
    
    // MARK: - Overloaded Functions (same name, different selectors)
    
    /// Selector: test(_:)
    /// Called as: test("hello")
    public func test(_ something: String) -> String {
        return "unlabeled: \(something)"
    }
    
    /// Selector: test(something:)
    /// Called as: test(something: "hello")
    public func test(something: String) -> String {
        return "labeled something: \(something)"
    }
    
    /// Selector: test(somethingElse:)
    /// Called as: test(somethingElse: "hello")
    public func test(somethingElse: String) -> String {
        return "labeled somethingElse: \(somethingElse)"
    }
    
    // MARK: - Caller Functions (to test CALLS edge accuracy)
    
    /// Calls test(_:) - the unlabeled overload.
    public func callUnlabeled() -> String {
        return test("from unlabeled caller")
    }
    
    /// Calls test(something:) - the 'something' labeled overload.
    public func callSomething() -> String {
        return test(something: "from something caller")
    }
    
    /// Calls test(somethingElse:) - the 'somethingElse' labeled overload.
    public func callSomethingElse() -> String {
        return test(somethingElse: "from somethingElse caller")
    }
    
    // MARK: - Multi-parameter Overloads
    
    /// Selector: configure(_:_:)
    public func configure(_ a: Int, _ b: Int) -> Int {
        return a + b
    }
    
    /// Selector: configure(width:height:)
    public func configure(width: Int, height: Int) -> Int {
        return width * height
    }
    
    /// Selector: configure(x:y:)
    public func configure(x: Int, y: Int) -> Int {
        return x - y
    }
    
    /// Calls configure(_:_:)
    public func callConfigureUnlabeled() -> Int {
        return configure(10, 20)
    }
    
    /// Calls configure(width:height:)
    public func callConfigureWidthHeight() -> Int {
        return configure(width: 100, height: 50)
    }
    
    /// Calls configure(x:y:)
    public func callConfigureXY() -> Int {
        return configure(x: 5, y: 3)
    }
}
