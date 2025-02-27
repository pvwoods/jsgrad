// Simple vanilla JavaScript unit tests for jsgrad

// Import the module with our classes
const { Value, Neuron, Layer, MLP, Conv2D } = require('./src/main.js');

// Set up console output capture for testing
const originalConsoleLog = console.log;
let consoleOutput = [];

// Capture console.log output for testing
console.log = function(...args) {
    consoleOutput.push(args.join(' '));
};

// Testing utilities
function assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

function assertAlmostEqual(a, b, epsilon = 1e-6, message) {
    if (Math.abs(a - b) > epsilon) {
        throw new Error(message || `Expected ${a} to be almost equal to ${b} (epsilon=${epsilon})`);
    }
}

// Tests for the Value class
function testValueClass() {
    console.log("Testing Value class...");
    
    // Test basic operations
    const a = new Value(2.0);
    const b = new Value(3.0);
    
    // Addition
    const c = a.add(b);
    assert(c.data === 5.0, "Addition failed");
    
    // Multiplication
    const d = a.mul(b);
    assert(d.data === 6.0, "Multiplication failed");
    
    // Subtraction
    const e = a.sub(b);
    assert(e.data === -1.0, "Subtraction failed");
    
    // Division
    const f = a.div(b);
    assertAlmostEqual(f.data, 2/3, 1e-6, "Division failed");
    
    // Exponentiation
    const g = a.pow(2);
    assert(g.data === 4.0, "Power operation failed");
    
    // Tanh
    const h = a.tanh();
    const expectedTanh = (Math.exp(2.0) - 1) / (Math.exp(2.0) + 1); // This matches the implementation in main.js
    assertAlmostEqual(h.data, expectedTanh, 1e-6, "Tanh failed");
    
    // Test gradient computation
    const x = new Value(2.0);
    const y = new Value(3.0);
    const z = x.mul(y).add(x); // z = x*y + x = 2*3 + 2 = 8
    
    // Do backward pass
    z.backward();
    
    // Check gradients
    assertAlmostEqual(x.grad, 4.0, 1e-6, "Gradient of x is incorrect"); // dx/dz = d(x*y + x)/dx = y + 1 = 3 + 1 = 4
    assertAlmostEqual(y.grad, 2.0, 1e-6, "Gradient of y is incorrect"); // dy/dz = d(x*y + x)/dy = x = 2
    
    console.log("Value class tests passed!");
    return true;
}

// Tests for the Conv2D class
function testConv2D() {
    console.log("Testing Conv2D class...");
    
    // Create a simple 3x3 input
    const input = [
        [new Value(1.0), new Value(2.0), new Value(3.0)],
        [new Value(4.0), new Value(5.0), new Value(6.0)],
        [new Value(7.0), new Value(8.0), new Value(9.0)]
    ];
    
    // Create a Conv2D with a single 2x2 filter with known weights
    const conv = new Conv2D(3, 3, 2, 1, 'test_conv');
    
    // Set filter weights manually for deterministic testing
    conv.filters[0][0] = new Value(0.5); // top-left
    conv.filters[0][1] = new Value(0.5); // top-right
    conv.filters[0][2] = new Value(0.5); // bottom-left
    conv.filters[0][3] = new Value(0.5); // bottom-right
    
    // Set bias to 0
    conv.biases[0] = new Value(0.0);
    
    // Output should be 2x2:
    // [0,0]: (1*0.5 + 2*0.5 + 4*0.5 + 5*0.5) = 6 -> tanh(6) 
    // [0,1]: (2*0.5 + 3*0.5 + 5*0.5 + 6*0.5) = 8 -> tanh(8)
    // [1,0]: (4*0.5 + 5*0.5 + 7*0.5 + 8*0.5) = 12 -> tanh(12)
    // [1,1]: (5*0.5 + 6*0.5 + 8*0.5 + 9*0.5) = 14 -> tanh(14)
    
    const output = conv.forward(input);
    
    // Check dimensions
    assert(output.length === 1, "Expected 1 channel in output");
    assert(output[0].length === 4, "Expected 4 values in output channel");
    
    // Check values (tanh will squash the values close to 1)
    // Using the same tanh implementation as in the main.js file
    const tanh6 = (Math.exp(6) - 1) / (Math.exp(6) + 1);
    const tanh8 = (Math.exp(8) - 1) / (Math.exp(8) + 1);
    const tanh12 = (Math.exp(12) - 1) / (Math.exp(12) + 1);
    const tanh14 = (Math.exp(14) - 1) / (Math.exp(14) + 1);
    
    assertAlmostEqual(output[0][0].data, tanh6, 1e-6, "Conv2D output [0,0] incorrect");
    assertAlmostEqual(output[0][1].data, tanh8, 1e-6, "Conv2D output [0,1] incorrect");
    assertAlmostEqual(output[0][2].data, tanh12, 1e-6, "Conv2D output [1,0] incorrect");
    assertAlmostEqual(output[0][3].data, tanh14, 1e-6, "Conv2D output [1,1] incorrect");
    
    // Create a much simpler test that doesn't rely on tanh activation
    // Create a simple 2x2 input
    const simpleInput = [
        [new Value(1.0), new Value(1.0)],
        [new Value(1.0), new Value(1.0)]
    ];
    
    // Create a simplified convolution with a 1x1 kernel (essentially just a single weight)
    // IMPORTANT: Don't replace the filter weights directly as it breaks the computational graph
    const simpleConv = new Conv2D(2, 2, 1, 1, 'simple_conv');
    
    // Forward pass
    const output2 = simpleConv.forward(simpleInput)[0];
    
    // Verify the output shape (should be 2x2)
    assert(output2.length === 4, "Expected 4 values in simple output");
    
    // With a randomly initialized filter, we can't check exact output values
    // But we can verify that the output contains valid Value objects
    for (let i = 0; i < 4; i++) {
        assert(output2[i] instanceof Value, `Output at position ${i} should be a Value object`);
    }
    
    // Create a simple MSE loss function directly without sum
    const target = new Value(1.0);
    const pred = output2[0]; // Just use the first output
    const loss = pred.sub(target).pow(2);
    
    // Print the computational graph information
    console.log("Loss op:", loss.op);
    console.log("Loss children count:", loss.children.length);
    
    // Log information about pred
    console.log("Pred op:", pred.op);
    console.log("Pred children count:", pred.children.length);
    
    // Check if the filter is in the computational graph
    let filterInGraph = false;
    
    function checkInGraph(node, depth = 0) {
        if (depth > 20) return; // Prevent infinite recursion
        
        if (node === simpleConv.filters[0][0]) {
            filterInGraph = true;
            return;
        }
        
        for (const child of node.children || []) {
            checkInGraph(child, depth + 1);
        }
    }
    
    checkInGraph(loss);
    console.log("Filter in graph:", filterInGraph);
    
    // Zero gradients
    simpleConv.filters[0][0].grad = 0;
    
    // Backward pass
    loss.backward();
    
    // Print gradient values for debugging
    console.log(`Filter gradient: ${simpleConv.filters[0][0].grad}`);
    
    // Now check that the gradient is non-zero
    assert(simpleConv.filters[0][0].grad !== 0, "Filter weight gradient should not be zero");
    
    console.log("Conv2D tests passed!");
    return true;
}

// Tests for the MLP class
function testMLP() {
    console.log("Testing MLP class...");
    
    // Create an MLP with 2 inputs, a hidden layer of 3 neurons, and 1 output
    const mlp = new MLP(2, [3, 1]);
    
    // Test forward pass
    const input = [new Value(1.0), new Value(2.0)];
    const output = mlp.forward(input);
    
    // Check output dimensions
    assert(output.length === 1, "Expected 1 output from MLP");
    assert(output[0] instanceof Value, "Output should be a Value instance");
    
    // Test parameter count
    const params = mlp.params();
    
    // Expected: 
    // Layer 1: 3 neurons, each with 2 weights + 1 bias = 3 * (2 + 1) = 9
    // Layer 2: 1 neuron with 3 weights + 1 bias = 4
    // Total: 9 + 4 = 13
    assert(params.length === 13, `Expected 13 parameters, got ${params.length}`);
    
    // Test zero_grad
    params.forEach(p => p.grad = 1.0);
    mlp.zero_grad();
    params.forEach(p => {
        assert(p.grad === 0, "zero_grad should reset all gradients to 0");
    });
    
    console.log("MLP tests passed!");
    return true;
}

// Run all tests
function runTests() {
    const tests = [
        testValueClass,
        testConv2D,
        testMLP
    ];
    
    let passed = 0;
    let failed = 0;
    
    for (const test of tests) {
        try {
            if (test()) {
                passed++;
            } else {
                failed++;
            }
        } catch (error) {
            console.error(`Test failed: ${error.message}`);
            console.error(error.stack);
            failed++;
        }
    }
    
    console.log(`\n--- Test Results ---`);
    console.log(`Passed: ${passed}`);
    console.log(`Failed: ${failed}`);
    
    return failed === 0;
}

// Restore original console.log
console.log = originalConsoleLog;

// Initialize and run tests
try {
    console.log("Running jsgrad unit tests...");
    const allPassed = runTests();
    console.log("All tests completed successfully!");
    process.exit(allPassed ? 0 : 1);
} catch (error) {
    console.error('Error running tests:', error);
    process.exit(1);
}