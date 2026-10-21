import sys

# Ensure UTF-8 output encoding for circuit diagram box-drawing characters on Windows
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

# Alias .qnn to .neural_network on TorchConnector for compatibility
if not hasattr(TorchConnector, "qnn"):
    TorchConnector.qnn = property(lambda self: self.neural_network)

def create_quantum_bottleneck(num_qubits=4):
    """Creates a Variational Quantum Circuit for the U-Net bottleneck."""
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    ansatz = RealAmplitudes(num_qubits, reps=1)
    qc = feature_map.compose(ansatz)
    
    # Measure Pauli-Z on each qubit so output dimension matches num_qubits (4)
    observables = [
        SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - 1 - i), 1)])
        for i in range(num_qubits)
    ]
    
    qnn = EstimatorQNN(
        circuit=qc,
        estimator=Estimator(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )
    
    return TorchConnector(qnn)

if __name__ == "__main__":
    q_layer = create_quantum_bottleneck(4)
    print("Quantum Bottleneck Layer Initialized Successfully!")
    print(q_layer.qnn.circuit)
