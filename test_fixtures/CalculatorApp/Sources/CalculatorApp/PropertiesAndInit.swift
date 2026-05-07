import Foundation

/// Phase 5 P0 fixture: exercises every detection path for the new
/// property/initializer/deinitializer extractors.
///
/// The class HouseholdAccount intentionally bundles together:
///   - stored properties (id, balance)
///   - simple-body computed property (displayName)
///   - explicit get/set computed property (formatted)
///   - lazy stored property (heavy)
///   - static stored property (sharedTaxRate)
///   - plain initializer (init(id:balance:))
///   - failable initializer (init?(legacyString:))
///   - deinitializer (deinit)
///
/// The struct Receipt covers stored + computed properties on a value
/// type, ensuring the extractor doesn't only fire inside class bodies.
public class HouseholdAccount {
    public let id: Int
    public var balance: Double
    public var displayName: String {
        return "Account #\(id)"
    }
    public var formatted: String {
        get { String(format: "%.2f", balance) }
        set { balance = Double(newValue) ?? 0 }
    }
    public lazy var heavy: String = {
        return "expensive: \(id)"
    }()
    public static var sharedTaxRate: Double = 0.07

    public init(id: Int, balance: Double) {
        self.id = id
        self.balance = balance
    }

    public init?(legacyString: String) {
        guard let parts = legacyString.split(separator: "|").map({ String($0) }).first else {
            return nil
        }
        self.id = Int(parts) ?? 0
        self.balance = 0.0
    }

    deinit {
        // cleanup hook
    }
}

public struct Receipt {
    public let total: Double
    public var taxIncluded: Double {
        return total * (1 + HouseholdAccount.sharedTaxRate)
    }
}
