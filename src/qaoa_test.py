from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

qp = QuadraticProgram(name="densest_k_subgraph")

for feature in ["f1", "f2", "f3", "f4"]:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={"f1": 1, "f2": 1, "f3": 1, "f4": 1},
    sense="==",
    rhs=2,
    name="select_k_features"
)

qp.maximize(
    quadratic={
        ("f1", "f2"): 0.9,
        ("f1", "f3"): 0.7,
        ("f2", "f3"): 0.8,
        ("f2", "f4"): 0.2,
        ("f3", "f4"): 0.3,
    }
)

algorithm_globals.random_seed = 42

sampler = StatevectorSampler(seed=42)
qaoa = QAOA(
    sampler=sampler,
    optimizer=COBYLA(),
    reps=2
)

optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)

print("Selected features:")
for name, value in zip(qp.variables, result.x):
    if value > 0.5:
        print(name.name)

print("Objective:", result.fval)
