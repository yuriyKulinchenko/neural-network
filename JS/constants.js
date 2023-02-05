
//the constants.js file will contain frequently used functions and constants

//constants

//multiplication of matrices and vectors

const multiply = function (matrix1, matrix2) {
    if (!(matrix2[0].constructor == Array)) {
        return multiplyVector(matrix1, matrix2);
    }
    let output = [];
    let sum = 0;
    for (let i = 0; i < matrix1.length; i++) {
        output.push([]);
    }
    for (let i = 0; i < matrix1.length; i++) {
        for (let j = 0; j < matrix2[0].length; j++) {
            sum = 0;
            for (let k = 0; k < matrix2.length; k++) {
                sum += matrix1[i][k] * matrix2[k][j];
            }
            output[i][j] = sum;
        }
    }
    if (output[0].length == 1) {
        matrixToVector(output);
    }
    return output;
}


const multiplyVector = function (matrix1, matrix2) {
    let sum = 0;
    let output = [];
    for (let i = 0; i < matrix1.length; i++) {
        sum = 0;
        for (let j = 0; j < matrix2.length; j++) {
            sum += matrix1[i][j] * matrix2[j];
        }
        output.push(sum);
    }
    return output;
}


const matrixToVector = function (matrix) {
    for (let i = 0; i < matrix.length; i++) {
        matrix[i] = matrix[i][0];
    }
}

const createVector = function (n, isRandom) {
    let a = new Array(n);
    for (let i = 0; i < n; i++) {
        if (isRandom) {
            a[i] = Math.random();
        } else {
            a[i] = 0;
        }
    }
    return a;
}

const createMatrix = function (n, m, isRandom) {
    let output = new Array(n);
    for (let i = 0; i < n; i++) {
        output[i] = createVector(m, isRandom);
    }
    return output;
}

const add = function (vector1, vector2) {
    let output = new Array(vector1.length);
    for (let i = 0; i < vector1.length; i++) {
        output[i] = vector1[i] + vector2[i];
    }
    return output;
}

//logistic functions

const sigmoid = function (x) {
    if (x.constructor == Array) {
        let output = [];
        for (let i = 0; i < x.length; i++) {
            output.push(sigmoid(x[i]));
        }
        return output;
    }
    let e = Math.E;
    return 1 / (1 + e ** (-x));
}

const sigmoidPrime = function (x) {
    if (x.constructor == Array) {
        let output = [];
        for (let i = 0; i < x.length; i++) {
            output.push(sigmoidPrime(x[i]));
        }
        return output;
    }
    let e = Math.E;
    return (e ** x) / ((1 + e ** x) ** 2);
}

const ReLU = function (x) {
    if (x.constructor == Array) {
        let output = [];
        for (let i = 0; i < x.length; i++) {
            output.push(ReLU(x[i]));
        }
        return output;
    }
    return Math.max(0, x);
}

const ReLUPrime = function (x) {
    if (x.constructor == Array) {
        let output = [];
        for (let i = 0; i < x.length; i++) {
            output.push(ReLUPrime(x[i]));
        }
        return output;
    }
    if (x > 0) {
        return 1;
    }
    return 0;
}

const ReLU6 = function (x) {
    if (x.constructor == Array) {
        let output = [];
        for (let i = 0; i < x.length; i++) {
            output.push(ReLU6(x[i]));
        }
        return output;
    }
    return Math.min(Math.max(0, x), 6);
}

const ReLU6Prime = function (x) {
    if (x.constructor == Array) {
        let output = [];
        for (let i = 0; i < x.length; i++) {
            output.push(ReLU6Prime(x[i]));
        }
        return output;
    }
    if (x <= 0 || x >= 6) {
        return 0;
    }
    return 1;
}


//cost functions

const cost = function (vector1, vector2) {
    let sum = 0;
    for (let i = 0; i < vector1.length; i++) {
        sum += (vector1[i] - vector2[i]) ** 2;
    }
    return sum;
}
