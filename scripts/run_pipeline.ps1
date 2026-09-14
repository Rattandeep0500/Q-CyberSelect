Write-Host "Q-CyberSelect Multi-Language Pipeline"
Write-Host ""
Write-Host "=== Python / Qiskit ==="
python ".\qiskit\experiments\qaoa_test.py"
Write-Host ""
Write-Host "=== C++ ==="
.\cpp\baselines\exact_dks.exe
Write-Host ""
Write-Host "=== Rust ==="
$env:Path += ";C:\msys64\ucrt64\bin;$HOME\.cargo\bin"
cargo run --manifest-path ".\rust\security_pipeline\Cargo.toml"
Write-Host ""
Write-Host "=== TypeScript ==="
if (!(Test-Path ".\frontend\dist\results.js")) {
    Set-Location ".\frontend"
    npx tsc --project .
    Set-Location ..
}
node ".\frontend\dist\results.js"
