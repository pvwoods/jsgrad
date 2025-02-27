class Value {

    constructor(data, children=null, op="", label="") {
        this.data = data;
        this.grad = 0;
        this.children = children != null ? children:[];
        this.op = op;
        this.label = label;
    }

    add(rhs) {
        if(!isNaN(rhs)) {
            rhs = new Value(rhs);
        }

        var result = new Value(this.data + rhs.data, [this, rhs], "+");

        var _back = () => {
            this.grad += 1.0 * result.grad;
            rhs.grad += 1.0 * result.grad;
        }

        result._backward = _back;
        
        return result;
    }

    neg() {
        return this.mul(-1);
    }

    sub(rhs) {
        if(!isNaN(rhs)) {
            rhs = new Value(-rhs)
        } else {
            rhs = rhs.neg();
        }
        return this.add(rhs);
    }

    mul(rhs) {
        if(!isNaN(rhs)) {
            rhs = new Value(rhs);
        }
        var result = new Value(this.data * rhs.data, [this, rhs], "*");

        var _back = () => {
            this.grad += rhs.data * result.grad;
            rhs.grad += this.data * result.grad;
        }

        result._backward = _back;

        return result;
    }

    div(rhs) {
        if(!isNaN(rhs)) {
            rhs = new Value(rhs);
        }
        var nrhs = rhs.pow(-1);
        return this.mul(nrhs);
    }

    tanh() {
        var value = (Math.exp(this.data) - 1) / (Math.exp(this.data) + 1);
        var result = new Value(value, [this], "tanh");

        var _back = () => {
            this.grad += (1 - value*value) * result.grad;
        }

        result._backward = _back;

        return result;
    }

    exp() {
        var value = Math.exp(this.data);
        var result = new Value(value, [this], "exp");

        var _back = () => {
            this.grad += value * result.grad;
        }

        result._backward = _back;

        return result;
    }

    pow(n) {
        var value = this.data ** n;
        var result = new Value(value, [this], "pow");

        var _back = () => {
            this.grad += (n * this.data ** (n - 1)) * result.grad;
        }

        result._backward = _back;

        return result;
    }

    _backward() {
        return;
    }

    backward() {
        // topo sort then call _backward
        var seen = {}
        var nodes = []

        var sort = (root) => {
            if(seen[root] == null) {
                seen[root] = true;
                for(var ci in root.children) {
                    sort(root.children[ci]);
                }
                nodes.push(root);
            }

        }
        sort(this);
        nodes = nodes.reverse();

        this.grad = 1.0;

        for(var ni in nodes) {
            nodes[ni]._backward();
        }
    }

    toString() {
        return `Value(data='${this.data}')`;
    }

}

// https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
function randn_bm() {
    let u = 1 - Math.random(); //Converting [0,1) to (0,1)
    let v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

class Neuron {
    constructor(tw, label='neuron') {
        this.weights = [];
        for(var w = 0; w < tw;w++) {
            this.weights.push(new Value(randn_bm(), null, '', `${label}:weight_${w}`));
        }
        this.bias = new Value(randn_bm(), null, '', `${label}:bias`);
    }

    forward(xs) {
        var sum = new Value(0.0);
        for(var i in this.weights) {
            var g = this.weights[i].mul(xs[i]);
            sum = sum.add(g);
        }
        sum = sum.add(this.bias);
        return sum.tanh();
    }

    params() {
        return [this.bias].concat(this.weights);
    }

}

class Conv2D {
    constructor(inputHeight, inputWidth, kernelSize, numFilters, label='conv') {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.kernelSize = kernelSize;
        this.numFilters = numFilters;
        
        // Output dimensions
        this.outputHeight = inputHeight - kernelSize + 1;
        this.outputWidth = inputWidth - kernelSize + 1;
        
        // Initialize filters (kernels)
        this.filters = [];
        for (let f = 0; f < numFilters; f++) {
            const filter = [];
            for (let i = 0; i < kernelSize; i++) {
                for (let j = 0; j < kernelSize; j++) {
                    filter.push(new Value(randn_bm() * 0.1, null, '', `${label}:filter_${f}_${i}_${j}`));
                }
            }
            this.filters.push(filter);
        }
        
        // Initialize biases
        this.biases = [];
        for (let f = 0; f < numFilters; f++) {
            this.biases.push(new Value(randn_bm() * 0.1, null, '', `${label}:bias_${f}`));
        }
    }
    
    forward(xs) {
        // xs should be a 2D array of Value objects with shape [inputHeight][inputWidth]
        const results = [];
        
        // For each filter
        for (let f = 0; f < this.numFilters; f++) {
            // For each position in the output
            const outputChannel = [];
            for (let i = 0; i <= this.inputHeight - this.kernelSize; i++) {
                for (let j = 0; j <= this.inputWidth - this.kernelSize; j++) {
                    // Apply convolution at this position
                    let filterIdx = 0;
                    
                    // Multiply and sum kernel values with input values
                    // Track all multiplications separately to ensure proper gradient flow
                    const weightedInputs = [];
                    for (let ki = 0; ki < this.kernelSize; ki++) {
                        for (let kj = 0; kj < this.kernelSize; kj++) {
                            const inputVal = xs[i + ki][j + kj];
                            // Explicitly create the multiplication operation
                            const weightedInput = inputVal.mul(this.filters[f][filterIdx]);
                            weightedInputs.push(weightedInput);
                            filterIdx++;
                        }
                    }
                    
                    // Sum all weighted inputs
                    const sum = weightedInputs.reduce((acc, val) => acc.add(val), new Value(0.0));
                    
                    // Add bias and apply activation
                    const withBias = sum.add(this.biases[f]);
                    const activated = withBias.tanh();
                    outputChannel.push(activated);
                }
            }
            results.push(outputChannel);
        }
        
        return results;
    }
    
    params() {
        let result = [...this.biases];
        for (let f = 0; f < this.numFilters; f++) {
            result = result.concat(this.filters[f]);
        }
        return result;
    }
}

class Layer {
    constructor(sin, sout) {
        this.neurons = []
        for(var s = 0; s < sout;s++) {
            this.neurons.push(new Neuron(sin, `neuron_${s}`));
        }
    }

    forward(xs) {
        var result = [];
        for(var ni in this.neurons) {
            result.push(this.neurons[ni].forward(xs));
        }
        return result;
    }

    params() {
        var result = [];
        for(var ni in this.neurons) {
            result = result.concat(this.neurons[ni].params());
        }
        return result;
    }
}

class MLP {
    constructor(sin, souts) {
        souts.unshift(sin)
        this.layers = []
        for(var s = 0; s < souts.length-1;s++) {
            this.layers.push(new Layer(souts[s], souts[s+1]));
        }
    }
    forward(xs) {
        for(var li in this.layers) {
            xs = this.layers[li].forward(xs);
        }
        return xs;
    }

    params() {
        var result = [];
        for(var li in this.layers) {
            result = result.concat(this.layers[li].params());
        }
        return result;
    }

    zero_grad() {
        var params = this.params();
        for(var pi in params) {
            params[pi].grad = 0;
        }
    }
}


// Run examples if in browser context (not being required by tests)
if (typeof window !== 'undefined') {
    // Example 1: MLP on a simple dataset
    let xs = [
        [ new Value(2.0), new Value(3.0), new Value(-1.0)],
        [ new Value(3.0), new Value(-1.0), new Value(0.5)],
        [ new Value(0.5), new Value(1.0), new Value(1.0)],
        [ new Value(1.0), new Value(1.0), new Value(-1.0)]
    ]
    let ys = [
        new Value(1.0),
        new Value(-1.0),
        new Value(-1.0),
        new Value(1.0)
    ]

    let model = new MLP(3, [4,4,1]);
    let lr = 0.2;
    let steps = 100;

    console.log("Training MLP model:");
    for(var step=0; step<steps;step++) {
        
        model.zero_grad();
        let loss = new Value(0.0);

        for(let ii in xs) {
            let pred = model.forward(xs[ii])[0];
            let diff = ys[ii].sub(pred).pow(2);
            loss = loss.add(diff);
        }
        loss = loss.div(ys.length);
        if(step % 10 == 0) {
            console.log(`step ${step}: ${loss.data}`);
        }
        loss.backward();

        var params = model.params();
        for(var pi in params) {
            let param = params[pi]
            param.data += param.grad * -lr;
        }
    }

    // Example 2: Using the 2D convolutional kernel
    console.log("\nTesting Conv2D on a 4x4 input:");

    // Create a 4x4 grid input
    const gridInput = [
        [new Value(1.0), new Value(0.0), new Value(0.0), new Value(1.0)],
        [new Value(0.0), new Value(1.0), new Value(1.0), new Value(0.0)],
        [new Value(0.0), new Value(1.0), new Value(1.0), new Value(0.0)],
        [new Value(1.0), new Value(0.0), new Value(0.0), new Value(1.0)]
    ];

    // Create a Conv2D layer with 2 filters of size 2x2
    const conv = new Conv2D(4, 4, 2, 2);

    // Forward pass
    const convOutput = conv.forward(gridInput);

    // Print the output shape
    console.log(`Conv2D output shape: ${convOutput.length} channels, ${convOutput[0].length} values per channel`);

    // Print the first few values from the output
    console.log("First channel output:");
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            console.log(`[${i},${j}]: ${convOutput[0][i*3+j].data.toFixed(4)}`);
        }
    }

    // Create a simple training example
    const targetOutput = [];
    for (let i = 0; i < 9; i++) {
        targetOutput.push(new Value(0.5));
    }

    // Simple training loop for the Conv2D
    console.log("\nTraining Conv2D for 5 steps:");
    for (let step = 0; step < 5; step++) {
        // Zero gradients
        const params = conv.params();
        for (let pi in params) {
            params[pi].grad = 0;
        }
        
        // Forward pass
        const output = conv.forward(gridInput)[0]; // Use only the first channel
        
        // Compute loss (MSE)
        let loss = new Value(0.0);
        for (let i = 0; i < output.length; i++) {
            const diff = output[i].sub(targetOutput[i]).pow(2);
            loss = loss.add(diff);
        }
        loss = loss.div(output.length);
        
        console.log(`step ${step}: ${loss.data.toFixed(6)}`);
        
        // Backward pass
        loss.backward();
        
        // Update parameters
        for (let pi in params) {
            params[pi].data += params[pi].grad * -0.1; // Smaller learning rate
        }
    }
}

// Export classes for testing in Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Value,
        Neuron,
        Layer,
        MLP,
        Conv2D
    };
}
