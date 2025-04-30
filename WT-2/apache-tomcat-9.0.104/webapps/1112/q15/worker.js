self.onmessage = function(event) {
    const num = parseInt(event.data);
    let results = [];

    function factorial(n) {
        return n === 0 ? 1 : n * factorial(n - 1);
    }

    for (let i = 1; i <= num; i++) {
        results.push(`${i}! = ${factorial(i)}`);
    }

    self.postMessage(results); // Send back factorial table
};
