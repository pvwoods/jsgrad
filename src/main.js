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
                    // CRITICAL FIX: Build the computational graph properly for gradient flow
                    let sum = null;
                    for (let ki = 0; ki < this.kernelSize; ki++) {
                        for (let kj = 0; kj < this.kernelSize; kj++) {
                            const inputVal = xs[i + ki][j + kj];
                            // Create the weighted input (multiplication)
                            const weightedInput = inputVal.mul(this.filters[f][filterIdx]);
                            
                            // Add to the running sum, ensuring proper graph connections
                            if (sum === null) {
                                sum = weightedInput;
                            } else {
                                sum = sum.add(weightedInput);
                            }
                            
                            filterIdx++;
                        }
                    }
                    
                    // Ensure we have a valid sum
                    if (sum === null) {
                        sum = new Value(0.0);
                    }
                    
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


// Examples are now controlled via the UI in index.html
// We won't automatically run them when the page loads

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
