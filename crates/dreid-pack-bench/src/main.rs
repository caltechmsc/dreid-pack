use std::ffi::OsStr;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use dreid_pack_bench::{AminoAcid, BenchConfig, BenchOutput, Format, Residue, ResidueTable};

const DEG: f64 = std::f64::consts::PI / 180.0;

type MetricFn = fn(&[Residue], f64) -> (u64, u64);

const METRICS: [(&str, MetricFn); 3] = [
    ("chi1", rate_chi1),
    ("chi1+2", rate_chi12),
    ("chi1-4", rate_chi14),
];

#[derive(Parser)]
#[command(name = "dpack-bench", version)]
struct Cli {
    /// Directory containing structure files (.cif or .pdb).
    dir: PathBuf,

    /// Emit CSV instead of a formatted table.
    #[arg(long)]
    csv: bool,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(&cli) {
        eprintln!("{} {e:#}", style("error:").red().bold());
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<()> {
    let entries = collect_entries(&cli.dir)?;
    let n = entries.len();

    if n == 0 {
        bail!("no .cif or .pdb files in '{}'", cli.dir.display());
    }

    let config = BenchConfig::default();

    let bar = ProgressBar::new(n as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "  {bar:40.cyan/dim}  {pos}/{len}  {wide_msg}  {elapsed_precise}",
        )
        .unwrap(),
    );

    let mut table = ResidueTable::new();
    let mut pack_time = Duration::ZERO;
    let mut ok = 0usize;
    let mut err = 0usize;

    let wall = Instant::now();

    for entry in &entries {
        let name = entry.file_stem().unwrap_or_default().to_string_lossy();
        bar.set_message(name.to_string());

        match bench_entry(entry, &config) {
            Ok(output) => {
                pack_time += output.elapsed;
                merge(&mut table, &output.table);
                ok += 1;
            }
            Err(e) => {
                bar.suspend(|| eprintln!("  {}  {name}: {e}", style("✗").red()));
                err += 1;
            }
        }

        bar.inc(1);
    }

    bar.finish_and_clear();

    if cli.csv {
        print_csv(&table);
    } else {
        print_table(&table, n, ok, err, pack_time, wall.elapsed());
    }

    Ok(())
}

fn bench_entry(path: &Path, config: &BenchConfig) -> Result<BenchOutput> {
    let fmt = format_from_ext(path.extension())?;
    let reader = BufReader::new(fs::File::open(path)?);
    Ok(dreid_pack_bench::bench(reader, fmt, config)?)
}

fn collect_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("cannot read '{}'", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(OsStr::to_str), Some("cif" | "pdb")))
        .collect();
    paths.sort();
    Ok(paths)
}

fn format_from_ext(ext: Option<&OsStr>) -> Result<Format> {
    match ext.and_then(OsStr::to_str) {
        Some("cif") => Ok(Format::Mmcif),
        Some("pdb") => Ok(Format::Pdb),
        _ => bail!("unsupported extension"),
    }
}

fn merge(dst: &mut ResidueTable, src: &ResidueTable) {
    for (aa, residues) in src.iter() {
        for r in residues {
            dst.push(aa, r.clone());
        }
    }
}

fn rate_chi1(residues: &[Residue], tol: f64) -> (u64, u64) {
    residues
        .iter()
        .filter_map(|r| r.chi_diff.first().and_then(|d| *d))
        .fold((0, 0), |(h, n), d| (h + (d <= tol) as u64, n + 1))
}

fn rate_chi12(residues: &[Residue], tol: f64) -> (u64, u64) {
    residues
        .iter()
        .filter(|r| r.chi_diff.len() >= 2 && r.chi_diff[..2].iter().all(|d| d.is_some()))
        .fold((0, 0), |(h, n), r| {
            let pass = r.chi_diff[0].unwrap() <= tol && r.chi_diff[1].unwrap() <= tol;
            (h + pass as u64, n + 1)
        })
}

fn rate_chi14(residues: &[Residue], tol: f64) -> (u64, u64) {
    residues
        .iter()
        .filter(|r| !r.chi_diff.is_empty() && r.chi_diff.iter().all(|d| d.is_some()))
        .fold((0, 0), |(h, n), r| {
            let pass = r.chi_diff.iter().all(|d| d.unwrap() <= tol);
            (h + pass as u64, n + 1)
        })
}

fn table_rate(table: &ResidueTable, f: MetricFn, tol: f64) -> (u64, u64) {
    table.iter().fold((0, 0), |(h, n), (_, rs)| {
        let (rh, rn) = f(rs, tol);
        (h + rh, n + rn)
    })
}

fn avg_rmsd_res(residues: &[Residue]) -> Option<f64> {
    let (sum, n) = residues
        .iter()
        .filter_map(|r| r.sc_rmsd)
        .fold((0.0, 0u64), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

fn avg_rmsd(table: &ResidueTable) -> Option<f64> {
    let (sum, n) = table
        .iter()
        .flat_map(|(_, rs)| rs.iter().filter_map(|r| r.sc_rmsd))
        .fold((0.0, 0u64), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

fn pct(hit: u64, tot: u64) -> f64 {
    if tot == 0 {
        0.0
    } else {
        hit as f64 / tot as f64 * 100.0
    }
}

fn fmt_rmsd(v: Option<f64>) -> String {
    v.map_or_else(|| format!("{:>8}", "—"), |v| format!("{v:>8.3}"))
}

fn print_table(
    table: &ResidueTable,
    total: usize,
    ok: usize,
    err: usize,
    pack_time: Duration,
    wall: Duration,
) {
    let packable: Vec<AminoAcid> = AminoAcid::ALL
        .iter()
        .copied()
        .filter(|aa| aa.n_chi() > 0)
        .collect();

    println!();
    println!(
        "  {}  {ok}/{total} structures  ({err} skipped)",
        style("Benchmark").bold(),
    );
    println!(
        "  {}  pack {:.2}s  wall {:.2}s",
        style("Timing").bold(),
        pack_time.as_secs_f64(),
        wall.as_secs_f64(),
    );
    println!();

    println!("  {:<12} {:>8} {:>8}", "", "20°", "40°");
    println!("  {}", "─".repeat(30));
    for (label, f) in METRICS {
        let (h20, n20) = table_rate(table, f, 20.0 * DEG);
        let (h40, n40) = table_rate(table, f, 40.0 * DEG);
        println!(
            "  {:<12} {:>7.1}% {:>7.1}%",
            style(label).bold(),
            pct(h20, n20),
            pct(h40, n40),
        );
    }
    println!("  {}", "─".repeat(30));
    println!(
        "  {:<12} {:>7.3} Å",
        style("RMSD").bold(),
        avg_rmsd(table).unwrap_or(0.0),
    );
    println!();

    println!("  Per-residue breakdown:");
    println!();
    println!(
        "  {:<4} {:>6}  {:>10} {:>10}  {:>8}",
        "AA", "N", "chi1-4 20°", "chi1-4 40°", "RMSD/Å"
    );
    let sep = format!("  {}", "─".repeat(44));
    println!("{sep}");

    for aa in &packable {
        let rs = &table[*aa];
        let (h20, t20) = rate_chi14(rs, 20.0 * DEG);
        let (h40, t40) = rate_chi14(rs, 40.0 * DEG);
        println!(
            "  {} {:>6}  {:>9.1}% {:>9.1}%  {}",
            style(format!("{:<4}", aa.code())).bold(),
            rs.len(),
            pct(h20, t20),
            pct(h40, t40),
            fmt_rmsd(avg_rmsd_res(rs)),
        );
    }

    let (h20, n20) = table_rate(table, rate_chi14, 20.0 * DEG);
    let (h40, n40) = table_rate(table, rate_chi14, 40.0 * DEG);
    let n_total: usize = packable.iter().map(|aa| table[*aa].len()).sum();
    println!("{sep}");
    println!(
        "  {} {:>6}  {:>9.1}% {:>9.1}%  {}",
        style(format!("{:<4}", "ALL")).bold(),
        n_total,
        pct(h20, n20),
        pct(h40, n40),
        fmt_rmsd(avg_rmsd(table)),
    );
    println!();
}

fn print_csv(table: &ResidueTable) {
    println!("aa,n,chi1_20,chi1_40,chi12_20,chi12_40,chi14_20,chi14_40,rmsd");
    for aa in AminoAcid::ALL.iter().filter(|a| a.n_chi() > 0) {
        let rs = &table[*aa];
        let (h1_20, n1_20) = rate_chi1(rs, 20.0 * DEG);
        let (h1_40, n1_40) = rate_chi1(rs, 40.0 * DEG);
        let (h12_20, n12_20) = rate_chi12(rs, 20.0 * DEG);
        let (h12_40, n12_40) = rate_chi12(rs, 40.0 * DEG);
        let (h14_20, n14_20) = rate_chi14(rs, 20.0 * DEG);
        let (h14_40, n14_40) = rate_chi14(rs, 40.0 * DEG);
        let rmsd = avg_rmsd_res(rs).map_or_else(String::new, |v| format!("{v:.4}"));
        println!(
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
            aa.code(),
            rs.len(),
            pct(h1_20, n1_20),
            pct(h1_40, n1_40),
            pct(h12_20, n12_20),
            pct(h12_40, n12_40),
            pct(h14_20, n14_20),
            pct(h14_40, n14_40),
            rmsd,
        );
    }
}
