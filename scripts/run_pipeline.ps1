Write-Host "Q-CyberSelect Multi-Language Pipeline"
python ".\qiskit\experiments\qaoa_test.py"
cargo run --manifest-path ".\rust\security_pipeline\Cargo.toml"
.\cpp\baselines\exact_dks.exe
node ".\frontend\dist\results.js"
