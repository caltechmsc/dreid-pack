use std::ffi::OsStr;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use dreid_pack_bench::{AminoAcid, BenchConfig, Format, Residue, ResidueTable};

const DEG: f64 = std::f64::consts::PI / 180.0;

type MetricFn = fn(&ResidueTable, f64) -> (u64, u64);

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

    let mut merged = ResidueTable::new();
    let mut pack_time = Duration::ZERO;
    let mut ok = 0usize;
    let mut err = 0usize;

    let wall = Instant::now();

    for entry in &entries {
        let name = entry.file_stem().unwrap_or_default().to_string_lossy();
        let fmt = format_from_ext(entry.extension());

        bar.set_message(name.to_string());

        let result = (|| -> Result<_> {
            let bytes = fs::read(entry)?;
            let reader = BufReader::new(std::io::Cursor::new(bytes));
            Ok(dreid_pack_bench::bench(reader, fmt?, &config)?)
        })();

        match result {
            Ok(output) => {
                pack_time += output.elapsed;
                merge(&mut merged, &output.table);
                ok += 1;
            }
            Err(e) => {
                bar.suspend(|| {
                    eprintln!("  {}  {name}: {e}", style("✗").red());
                });
                err += 1;
            }
        }

        bar.inc(1);
    }

    bar.finish_and_clear();

    if cli.csv {
        print_csv(&merged);
    } else {
        print_table(&merged, n, ok, err, pack_time, wall.elapsed());
    }

    Ok(())
}

fn collect_entries(dir: &PathBuf) -> Result<Vec<PathBuf>> {
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
    let mut hit = 0u64;
    let mut tot = 0u64;
    for r in residues {
        if let Some(Some(d)) = r.chi_diff.first() {
            tot += 1;
            if *d <= tol {
                hit += 1;
            }
        }
    }
    (hit, tot)
}

fn rate_chi12(residues: &[Residue], tol: f64) -> (u64, u64) {
    let mut hit = 0u64;
    let mut tot = 0u64;
    for r in residues {
        if r.chi_diff.len() >= 2 && r.chi_diff[0].is_some() && r.chi_diff[1].is_some() {
            tot += 1;
            if r.chi_diff[0].unwrap() <= tol && r.chi_diff[1].unwrap() <= tol {
                hit += 1;
            }
        }
    }
    (hit, tot)
}

fn rate_chi14(residues: &[Residue], tol: f64) -> (u64, u64) {
    let mut hit = 0u64;
    let mut tot = 0u64;
    for r in residues {
        if r.chi_diff.is_empty() || r.chi_diff.iter().any(|d| d.is_none()) {
            continue;
        }
        tot += 1;
        if r.chi_diff.iter().all(|d| d.unwrap() <= tol) {
            hit += 1;
        }
    }
    (hit, tot)
}

fn avg_rmsd(table: &ResidueTable) -> (f64, u64) {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for (_aa, residues) in table.iter() {
        for r in residues {
            if let Some(v) = r.sc_rmsd {
                sum += v;
                n += 1;
            }
        }
    }
    (sum, n)
}

fn avg_rmsd_res(residues: &[Residue]) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for r in residues {
        if let Some(v) = r.sc_rmsd {
            sum += v;
            n += 1;
        }
    }
    if n > 0 { Some(sum / n as f64) } else { None }
}

fn pct(hit: u64, tot: u64) -> f64 {
    if tot == 0 {
        0.0
    } else {
        hit as f64 / tot as f64 * 100.0
    }
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

    let metrics: [(&str, MetricFn); 3] = [
        ("chi1   ", |t: &ResidueTable, tol| {
            let mut h = 0u64;
            let mut n = 0u64;
            for (_, rs) in t.iter() {
                let (a, b) = rate_chi1(rs, tol);
                h += a;
                n += b;
            }
            (h, n)
        }),
        ("chi1+2 ", |t: &ResidueTable, tol| {
            let mut h = 0u64;
            let mut n = 0u64;
            for (_, rs) in t.iter() {
                let (a, b) = rate_chi12(rs, tol);
                h += a;
                n += b;
            }
            (h, n)
        }),
        ("chi1-4 ", |t: &ResidueTable, tol| {
            let mut h = 0u64;
            let mut n = 0u64;
            for (_, rs) in t.iter() {
                let (a, b) = rate_chi14(rs, tol);
                h += a;
                n += b;
            }
            (h, n)
        }),
    ];

    println!("  {:<12} {:>8} {:>8}", "", "20°", "40°");
    println!("  {}", "─".repeat(30));
    for (label, f) in &metrics {
        let (h20, n20) = f(table, 20.0 * DEG);
        let (h40, n40) = f(table, 40.0 * DEG);
        println!(
            "  {:<12} {:>7.1}% {:>7.1}%",
            style(label).bold(),
            pct(h20, n20),
            pct(h40, n40),
        );
    }

    let (rmsd_sum, rmsd_n) = avg_rmsd(table);
    println!("  {}", "─".repeat(30));
    println!(
        "  {:<12} {:>7.3} Å",
        style("RMSD").bold(),
        if rmsd_n > 0 {
            rmsd_sum / rmsd_n as f64
        } else {
            0.0
        },
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

    let mut h_all_20 = 0u64;
    let mut n_all_20 = 0u64;
    let mut h_all_40 = 0u64;
    let mut n_all_40 = 0u64;
    let mut n_total = 0usize;

    for aa in &packable {
        let rs = &table[*aa];
        n_total += rs.len();
        let (h20, t20) = rate_chi14(rs, 20.0 * DEG);
        let (h40, t40) = rate_chi14(rs, 40.0 * DEG);
        h_all_20 += h20;
        n_all_20 += t20;
        h_all_40 += h40;
        n_all_40 += t40;
        let rmsd_str = match avg_rmsd_res(rs) {
            Some(v) => format!("{v:>8.3}"),
            None => format!("{:>8}", "—"),
        };
        println!(
            "  {} {:>6}  {:>9.1}% {:>9.1}%  {}",
            style(format!("{:<4}", aa.code())).bold(),
            rs.len(),
            pct(h20, t20),
            pct(h40, t40),
            rmsd_str,
        );
    }

    println!("{sep}");
    let rmsd_all_str = if rmsd_n > 0 {
        format!("{:>8.3}", rmsd_sum / rmsd_n as f64)
    } else {
        format!("{:>8}", "—")
    };
    println!(
        "  {} {:>6}  {:>9.1}% {:>9.1}%  {}",
        style(format!("{:<4}", "ALL")).bold(),
        n_total,
        pct(h_all_20, n_all_20),
        pct(h_all_40, n_all_40),
        rmsd_all_str,
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
        let rmsd = match avg_rmsd_res(rs) {
            Some(v) => format!("{v:.4}"),
            None => String::new(),
        };
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
