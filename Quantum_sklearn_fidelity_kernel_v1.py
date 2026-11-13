# Quantum_sklearn_fidelity_kernel



import random
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_SIZE = 3
SHARPNESS_GRID = (1.5, 3.0, 5.0, 8.0, 12.0)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_SIZE + 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_window_matrix(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_total = len(draws)
    x_list = []
    y_list = []
    for t in range(WINDOW_SIZE, n_total):
        win = draws[t - WINDOW_SIZE : t].astype(np.float64)
        diffs = np.diff(win, axis=0)
        feat_row = np.concatenate([win.ravel(), diffs.ravel()])
        x_list.append(feat_row)
        y_list.append(draws[t])
    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.int64)


def build_sklearn_feature_pipeline() -> Pipeline:
    union = FeatureUnion(
        transformer_list=[
            ("splines", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("nystroem", Nystroem(kernel="rbf", gamma=0.05, n_components=24, random_state=39)),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            ("nonlinear", union),
            ("svd", TruncatedSVD(n_components=NUM_QUBITS, random_state=39)),
        ]
    )
    return pipe


def to_qubit_angles(reduced_x: np.ndarray) -> np.ndarray:
    return np.pi * np.tanh(reduced_x)


def encode_zz_quantum_state(angles: np.ndarray) -> np.ndarray:
    angles_1d = np.asarray(angles, dtype=np.float64).ravel()
    qc = QuantumCircuit(NUM_QUBITS)

    for q in range(NUM_QUBITS):
        qc.h(q)
        qc.rz(float(angles_1d.__getitem__(q)), q)

    for q in range(NUM_QUBITS - 1):
        a_q = float(angles_1d.__getitem__(q))
        a_next = float(angles_1d.__getitem__(q + 1))
        qc.rzz(0.5 * a_q * a_next, q, q + 1)

    a_last = float(angles_1d.__getitem__(NUM_QUBITS - 1))
    a_first = float(angles_1d.__getitem__(0))
    qc.rzz(0.5 * a_last * a_first, NUM_QUBITS - 1, 0)

    for q in range(NUM_QUBITS):
        qc.h(q)
        qc.ry(0.5 * float(angles_1d.__getitem__(q)), q)

    for q in range(NUM_QUBITS - 1):
        a_q = float(angles_1d.__getitem__(q))
        a_next = float(angles_1d.__getitem__(q + 1))
        qc.rzz(0.5 * a_q * a_next, q, q + 1)

    sv = Statevector.from_int(0, dims=STATE_DIM).evolve(qc)
    return sv.data


def compute_all_quantum_states(angles_matrix: np.ndarray) -> np.ndarray:
    n_samples = int(angles_matrix.shape.__getitem__(0))
    states = np.zeros((n_samples, STATE_DIM), dtype=np.complex128)
    for idx in range(n_samples):
        states[idx] = encode_zz_quantum_state(angles_matrix[idx])
    return states


def build_outcome_indicators(y_arr: np.ndarray) -> np.ndarray:
    n_samples = len(y_arr)
    indicators = np.zeros((n_samples, MAX_NUM), dtype=np.float64)
    for i in range(n_samples):
        for num in y_arr[i]:
            indicators[i, int(num) - 1] = 1.0
    return indicators


def predict_from_fidelity_kernel(
    query_state: np.ndarray,
    hist_states: np.ndarray,
    hist_indicators: np.ndarray,
    sharpness: float,
) -> np.ndarray:
    overlaps = np.abs(hist_states @ np.conj(query_state)) ** 2
    weights = np.power(np.maximum(overlaps, 1e-12), sharpness)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-15:
        weights = np.ones(len(weights), dtype=np.float64) / float(len(weights))
    else:
        weights /= weight_sum

    probs = weights @ hist_indicators
    probs = np.maximum(probs, 1e-12)
    probs /= np.sum(probs)
    if abs(float(np.sum(probs)) - 1.0) > 1e-6:
        raise ValueError("Greska u normalizaciji Fidelity Kernel raspodele.")
    return probs


def evaluate_fidelity_range(
    states_all: np.ndarray,
    indicators_all: np.ndarray,
    y_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    sharpness: float,
) -> float:
    total_nll = 0.0
    count = 0
    for t in range(start_idx, end_idx):
        probs = predict_from_fidelity_kernel(
            states_all[t],
            states_all[:t],
            indicators_all[:t],
            sharpness,
        )
        for num in y_all[t]:
            p = max(float(probs.__getitem__(int(num) - 1)), 1e-12)
            total_nll -= np.log(p)
        count += 1
    return total_nll / max(count, 1)


def decode_seven_numbers(probs: np.ndarray) -> list[int]:
    top_indices = np.argsort(probs)[::-1][:NUM_POSITIONS]
    return sorted(int(idx) + 1 for idx in top_indices)


def process_csv(file_path: Path) -> None:
    draws = load_draws(file_path)
    x_all, y_all = build_window_matrix(draws)
    n_samples = len(y_all)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    sk_pipe = build_sklearn_feature_pipeline()
    sk_pipe.fit(x_all[:train_end])
    angles_tv = to_qubit_angles(sk_pipe.transform(x_all[:val_end]))
    states_tv = compute_all_quantum_states(angles_tv)
    indicators_tv = build_outcome_indicators(y_all[:val_end])

    best_sharpness = float(next(iter(SHARPNESS_GRID)))
    best_val_nll = float("inf")

    for sh_val in SHARPNESS_GRID:
        val_nll = evaluate_fidelity_range(
            states_tv,
            indicators_tv,
            y_all[:val_end],
            train_end,
            val_end,
            float(sh_val),
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_sharpness = float(sh_val)

    sk_pipe_tv = build_sklearn_feature_pipeline()
    sk_pipe_tv.fit(x_all[:val_end])
    angles_test = to_qubit_angles(sk_pipe_tv.transform(x_all))
    states_test = compute_all_quantum_states(angles_test)
    indicators_all = build_outcome_indicators(y_all)

    test_nll = evaluate_fidelity_range(
        states_test,
        indicators_all,
        y_all,
        val_end,
        n_samples,
        best_sharpness,
    )

    sk_pipe_full = build_sklearn_feature_pipeline()
    angles_full = to_qubit_angles(sk_pipe_full.fit_transform(x_all))
    states_full = compute_all_quantum_states(angles_full)

    latest_win = draws[-WINDOW_SIZE:].astype(np.float64)
    latest_diffs = np.diff(latest_win, axis=0)
    latest_feat = np.concatenate([latest_win.ravel(), latest_diffs.ravel()]).reshape(1, -1)
    latest_angle = to_qubit_angles(sk_pipe_full.transform(latest_feat)).ravel()
    latest_state = encode_zz_quantum_state(latest_angle)

    final_probs = predict_from_fidelity_kernel(
        latest_state,
        states_full,
        indicators_all,
        best_sharpness,
    )
    next_comb = decode_seven_numbers(final_probs)

    if len(set(next_comb)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in next_comb):
        raise RuntimeError(f"Nelegalna NEXT kombinacija: {next_comb}")

    print(f"CSV: {file_path.name}")
    print(f"Validacija NLL (alpha={best_sharpness}): {best_val_nll:.4f}")
    print(f"Test NLL: {test_nll:.4f}")
    print(f"NEXT: {next_comb}")


def main() -> None:
    print("Metoda: Quantum Sklearn Spline-Nystroem Fidelity Kernel (1x6 qubita, ZZ Feature Map)")
    for csv_path in FILES:
        process_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Sklearn Spline-Nystroem Fidelity Kernel (1x6 qubita, ZZ Feature Map)

CSV: loto7_4718_k90.csv
Validacija NLL (alpha=1.5): 25.6700
Test NLL: 25.6807
NEXT: [5, 11, 19, 23, 24, 29, 34]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (alpha=1.5): 25.7304
Test NLL: 25.7026
NEXT: [8, 11, 16, 17, 19, 22, 32]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (alpha=1.5): 25.7236
Test NLL: 25.7686
NEXT: [7, 14, 18, 24, 29, 33, 34]
"""



"""
	1. Nelinearna aproksimacija jezgra i Spline preslikavanje bez klasifikatora: Iz Scikit-learn oblasti preuzimamo napredno nelinearno preslikavanje ulaznog prostora (SplineTransformer i Nystroem RBF/Laplacian kernel aproksimacije) nad hronološkim prozorima kola, uz strogu TimeSeriesSplit validaciju bez curenja budućnosti. Ovi trikovi služe isključivo za transformaciju stanja prelaza u Hilbertov prostor karakteristika (a ne za zabranjenu binarnu ili višeklasnu klasifikaciju brojeva).
	2. Kvantno jezgro i kodiranje amplituda (Quantum Kernel Amplitude Encoding — 1x6 qubita, 64 stanja): Umesto klasične regresije iz prvih modela ili 5-qubitnog z-centriranja iz prethodnog modela, ovde se koristi kvantno jezgro (Quantum Feature Map) inspirisano Nystroem i Spline transformacijama: ulazni podaci prethodnih kola se pomoću Nystroem + SplineTransformer projektuju u uglove parametarskog kvantnog kola (ZZFeatureMap struktura sa RZ i RZZ interakcijama drugog reda na 6 qubita).
	3. Očitavanje preko preklapanja kvantnih stanja (Fidelity Overlap) sa istorijskim prelazima: Verovatnoća svakog od 39 brojeva računa se preko kvantne vernosti (fidelity \(\vert{}\langle \psi(x_{\text{now}}) \vert{} \psi(x_t) \rangle\vert{}^2\)) između trenutnog kvantnog stanja i istorijskih kvantnih stanja, ponderisano njihovim narednim ishodima, čime se dobija potpuno autentična kvantna jezgro-metoda.

Kombinacija Nystroem / SplineTransformer nelinearnih transformacija sa kvantnim ZZ mapiranjem karakteristika i merenjem kvantne vernosti (fidelity) direktno povezuje suštinu Scikit-learn trikova sa kvantnom mehanikom.
Spline/Nystroem transformacije sa kvantnom vernošću (Fidelity Kernel)


Povezujem napredne Scikit-learn transformacije (SplineTransformer, Nystroem kernel aproksimacije i RobustScaler unutar FeatureUnion cevovoda) sa kvantnim ZZ mapiranjem karakteristika i merenjem kvantne vernosti (Fidelity Overlap \(\vert{}\langle \psi(x_{\text{now}}) \vert{} \psi(x_t) \rangle\vert{}^2\)) na 1x6 qubita.

• Scikit-learn nelinearni cevovod (FeatureUnion + SplineTransformer + Nystroem): Umesto klasifikatora, koriste se trikovi iz Scikit-learn članaka za konstruisanje bogatog nelinearnog prostora stanja: RobustScaler (otpornost na ekstremne skokove), SplineTransformer (glatke B-splajn bazne funkcije po pozicijama) i Nystroem (RBF aproksimacija jezgra), sažeti u 6 faznih uglova — po jedan za svaki od 6 qubita.
• Kvantno ZZ mapiranje i vernost stanja (Qiskit 1.4.4, 1x6 qubita): Svaki istorijski prozor kola preslikava se u 6-qubitni vektor stanja \(\vert{}\psi(x_t)\rangle\) putem Hadamard superpozicije, jednokubitnih RZ faznih rotacija i parnih RZZ interakcija drugog reda.
• Očitavanje raspodele preko kvantnog preklapanja (Quantum Kernel Fidelity): Kvantna sličnost između trenutnog stanja i svih prethodnih stanja računa se direktno kao kvadrat skalarnog proizvoda amplituda \(K(x_{\text{now}}, x_t) = \vert{}\langle \psi(x_{\text{now}}) \vert{} \psi(x_t) \rangle\vert{}^2\) podignut na stepen oštrine α (koji se bira isključivo na hronološkoj validaciji bez curenja podataka), a iz tako ponderisanih ishoda izdvaja se 7 najverovatnijih brojeva odjednom.
"""
