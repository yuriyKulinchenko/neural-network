class Network {
    constructor(layers, cost, logistic, logisticPrime, batchSize, learnRate) {
        this.cost = cost;
        this.logistic = logistic;
        this.logisticPrime = logisticPrime;
        this.batchSize = batchSize;
        this.a = this.generateA(layers);
        this.b = this.generateA(layers);
        this.z = this.generateA(layers);
        this.w = this.generateW(layers);
        this.aOutput = null;

        this.da = this.generateA(layers);
        this.db = this.generateA(layers);
        this.dw = this.generateW(layers);

        this.daAverage = this.generateA(layers);
        this.dbAverage = this.generateA(layers);
        this.dwAverage = this.generateW(layers);

        //these values are never referenced
        this.da[0] = null;
        this.db[0] = null;

        this.daAverage[0] = null;
        this.dbAverage[0] = null;

        this.b[0] = null;
        this.z[0] = null;

        this.depth = this.a.length;
        this.learnRate = learnRate;

        this.resetAverage();


        //'da' represents the partial derivative of the cost function with respect to a
    }
    generateA(layers) {
        let output = [];
        for (let i = 0; i < layers.length; i++) {
            output.push(createVector(layers[i], true));
        }
        return output;
    }
    generateW(layers) {
        let output = [null];
        for (let i = 1; i < layers.length; i++) {
            output.push(createMatrix(layers[i], layers[i - 1], true));
        }
        return output;
    }
    input(vector) {//expected output = array
        this['a'][0] = vector;
        for (let l = 1; l < this['a'].length; l++) {
            this['z'][l] = add(multiply(this['w'][l], this['a'][l - 1]), this['b'][l]);
            this['a'][l] = this.logistic(this['z'][l]);
        }
        return this['a'][this['a'].length - 1];
    }

    trainBatch() {//this takes dataset.training 
        let set = mnist.set(this.batchSize, 10);
        let trainingSet = set.training;
        let testSet = set.test;
        let currentTestcase;
        this.resetAverage();
        for (let i = 0; i < this.batchSize; i++) {
            currentTestcase = trainingSet[i];
            this.train(currentTestcase.input, currentTestcase.output);
            this.updateAverage();
        }
        this.updateNetwork();
        for (let i = 0; i < 1; i++) {
            currentTestcase = testSet[i];
            this.input(currentTestcase.input);
            console.log(this.cost(this.a[this.depth - 1], currentTestcase.output));
        }
    }

    train(testCase, expectedOutput) {
        this.input(testCase);
        for (let L = this.depth - 1; L > 0; L--) {//going down through the layers
            //first, find da
            for (let i = 0; i < this.da[L].length; i++) {
                this.da[L][i] = this.diffA(L, i, expectedOutput);
                if (isNaN(this.da[L][i])) { console.log(`error: activation L:${L} i:${i} is not defined`) };
            }
            //next, find db
            for (let i = 0; i < this.db[L].length; i++) {
                this.db[L][i] = this.diffB(L, i);
                if (isNaN(this.db[L][i])) { console.log(`error: bias L:${L} i:${i} is not defined`) };
            }
            //next, find dw
            //unlike da or db, dw is a matrix
            for (let i = 0; i < this.dw[L].length; i++) {
                for (let j = 0; j < this.dw[L][0].length; j++) {
                    this.dw[L][i][j] = this.diffW(L, i, j);
                    if (isNaN(this.dw[L][i][j])) { console.log(`error: weight L:${L} i:${i} j:${j} is not defined`) };
                }
            }
        }
    }

    diffA(L, i, y = null) {
        //y is the expected output
        //the expected output is needed if L is the last layer
        let [a, w, db] = [this.a, this.w, this.db];
        if (L == this.depth - 1) {
            return 2 * (a[L][i] - y[i]);
        }
        let length = db[L + 1].length;
        let sum = 0;
        for (let j = 0; j < length; j++) {
            sum += db[L + 1][j] * w[L + 1][j][i];
        }
        return sum;
    }

    diffB(L, i) {
        let [da, z] = [this.da, this.z];
        let logisticPrime = this.logisticPrime;
        return da[L][i] * logisticPrime(z[L][i]);
    }

    diffW(L, i, j) {
        let [db, a] = [this.db, this.a];
        return db[L][i] * a[L - 1][j];
    }

    resetAverage() {//changes average values back to 0
        for (let L = 1; L < this.depth; L++) {
            for (let i = 0; i < this.daAverage[L].length; i++) {
                this.daAverage[L][i] = 0;
                this.dbAverage[L][i] = 0;
                for (let j = 0; j < this.dwAverage[L][0].length; j++) {
                    this.dwAverage[L][i][j] = 0;
                }
            }

        }
    }
    updateAverage() {
        let learnRate = this.learnRate;
        for (let L = 1; L < this.depth; L++) {
            for (let i = 0; i < this.daAverage[L].length; i++) {
                this.daAverage[L][i] += this.da[L][i] * learnRate / this.batchSize;
                this.dbAverage[L][i] += this.db[L][i] * learnRate / this.batchSize;
                for (let j = 0; j < this.dwAverage[L][0].length; j++) {
                    this.dwAverage[L][i][j] += this.dw[L][i][j] * learnRate / this.batchSize;
                }
            }

        }
    }
    updateNetwork() {
        for (let L = 1; L < this.depth; L++) {
            for (let i = 0; i < this.b[L].length; i++) {
                this.b[L][i] -= this.dbAverage[L][i];
                for (let j = 0; j < this.dwAverage; j++) {
                    this.w[L][i][j] -= this.dwAverage[L][i][j];
                }
            }
        }
    }

}

let network = new Network([784, 10, 10], cost, sigmoid, sigmoidPrime, 1000, 1);

function run(network) {
    for (let i = 0; i < 1000; i++) {
        network.trainBatch();
    }
}
