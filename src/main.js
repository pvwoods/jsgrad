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

for(var step=0; step<steps;step++) {
    
    model.zero_grad();
    loss = new Value(0.0);

    for(ii in xs) {
        pred = model.forward(xs[ii])[0];
        diff = ys[ii].sub(pred).pow(2);
        loss = loss.add(diff);
    }
    loss = loss.div(ys.length);
    if(step % 10 == 0) {
        console.log(`step ${step}: ${loss.data}`);
    }
    loss.backward();

    var params = model.params();
    for(var pi in params) {
        param = params[pi]
        param.data += param.grad * -lr;
    }
}
