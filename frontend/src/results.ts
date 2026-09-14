const results = {
    qaoa: {
        features: 3,
        accuracy: 0.9167,
        f1: 0.9165
    },
    classical_mi: {
        features: 3,
        accuracy: 0.9167,
        f1: 0.9165
    },
    all_features: {
        features: 13,
        accuracy: 0.9722,
        f1: 0.9720
    }
};

console.log("Q-CyberSelect Results");
console.log("=====================");

for (const [method, result] of Object.entries(results)) {
    console.log(
        `${method}: ${result.features} features | ` +
        `accuracy=${(result.accuracy * 100).toFixed(2)}% | ` +
        `F1=${(result.f1 * 100).toFixed(2)}%`
    );
}
