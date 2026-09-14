use std::fs;

fn main() {
    let path = "data/sample_network.csv";
    let content = fs::read_to_string(path).expect("failed to read file");
    let mut lines = content.lines();

    let header = lines.next().unwrap_or("");
    let columns = header.split(',').count();
    let rows = lines.count();

    println!("Rust Security Pipeline");
    println!("Columns: {}", columns);
    println!("Rows: {}", rows);
}
