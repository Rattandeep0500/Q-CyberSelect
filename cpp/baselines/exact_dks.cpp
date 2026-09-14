#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

int main() {
    vector<string> features = {
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols"
    };

    double W[6][6] = {
        {1.0000, 0.0953, 0.2103, 0.3102, 0.3120, 0.2898},
        {0.0953, 1.0000, 0.1640, 0.2891, 0.0543, 0.3350},
        {0.2103, 0.1640, 1.0000, 0.4434, 0.2866, 0.1280},
        {0.3102, 0.2891, 0.4434, 1.0000, 0.0986, 0.1441},
        {0.3120, 0.0543, 0.2866, 0.0986, 1.0000, 0.2149},
        {0.2898, 0.3350, 0.1280, 0.1441, 0.2149, 1.0000}
    };

    int best[6] = {};
    double best_score = -1.0;

    for (int a = 0; a < 6; ++a) {
        for (int b = a + 1; b < 6; ++b) {
            for (int c = b + 1; c < 6; ++c) {
                double score = W[a][b] + W[a][c] + W[b][c];

                if (score > best_score) {
                    best_score = score;

                    for (int i = 0; i < 6; ++i)
                        best[i] = 0;

                    best[a] = 1;
                    best[b] = 1;
                    best[c] = 1;
                }
            }
        }
    }

    cout << "C++ Exact Densest-k-Subgraph" << endl;
    cout << "Selected:" << endl;

    for (int i = 0; i < 6; ++i) {
        if (best[i])
            cout << features[i] << endl;
    }

    cout << "Objective: " << best_score << endl;

    return 0;
}
