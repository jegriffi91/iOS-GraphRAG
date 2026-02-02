// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CalculatorApp",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "CalculatorApp", targets: ["CalculatorApp"])
    ],
    targets: [
        .target(name: "CalculatorApp")
    ]
)
