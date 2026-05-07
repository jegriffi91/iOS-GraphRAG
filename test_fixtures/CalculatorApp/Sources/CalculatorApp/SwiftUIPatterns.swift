import SwiftUI
import Combine

/// Phase 4.5d fixture: exercises every SwiftUI/SwiftData/Combine pattern the
/// extractor recognizes. One file packs all of:
///
///   - is_observable=TRUE via @Observable on a class (AuthState).
///   - is_swiftui_view=TRUE via View conformance (LoginView, ProfileBadge).
///   - state_kind on each supported wrapper attribute (@StateObject, @State,
///     @Environment, @AppStorage, @ObservedObject, @Binding, @Published).
///   - body_kind='viewbody' on the `body: some View` computed property.
///   - body_kind='resultbuilder' on a @ViewBuilder function.
///   - is_observable on a class with @Published (Counter) -- must remain FALSE
///     because the @Published wrapper is property-level, not class-level.

@Observable
class AuthState {
    var username: String = ""
    var isAuthenticated: Bool = false
}

struct LoginView: View {
    @StateObject var auth = AuthState()
    @State private var password: String = ""
    @Environment(\.colorScheme) var colorScheme
    @AppStorage("rememberMe") var remember: Bool = false

    var body: some View {
        Text("Hello")
    }
}

struct ProfileBadge: View {
    @ObservedObject var account: AuthState
    @Binding var isVisible: Bool

    var body: some View {
        Text(account.username)
    }
}

@ViewBuilder
func makeButtonRow() -> some View {
    Text("Button row")
}

class Counter {
    @Published var count: Int = 0
}
