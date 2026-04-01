use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use dreid_pack::Progress;

fn pending_style() -> ProgressStyle {
    ProgressStyle::with_template("  ·  {msg}").unwrap()
}

fn running_style() -> ProgressStyle {
    ProgressStyle::with_template("  {spinner:.cyan}  {msg}").unwrap()
}

fn done_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}").unwrap()
}

const BANNER: &str = r#"
________________________________________     ________             ______
___  __ \__  __ \__  ____/___  _/__  __ \    ___  __ \_____ _________  /__
__  / / /_  /_/ /_  __/   __  / __  / / /    __  /_/ /  __ `/  ___/_  //_/
_  /_/ /_  _, _/_  /___  __/ /  _  /_/ /     _  ____// /_/ // /__ _  ,<
/_____/ /_/ |_| /_____/  /___/  /_____/      /_/     \__,_/ \___/ /_/|_|
─────────────── Full-Atom Side-Chain Packing with DREIDING ───────────────
    H    CH₃      H   H   O     CH₂SH     H   H   O   (CH₂)₄NH₂ O
     \    │       │   │   ║       │       │   │   ║       │    ⫽
      N───C───C───N───C───C───N───C───C───N───C───C───N───C───C
     /    │   ║       │       │   │   ║       │       │        \
    H     H   O     CH₂OH     H   H   O    CH₂COOH    H         O─H
"#;

pub fn banner() -> String {
    let mut out = String::new();
    for (i, line) in BANNER.lines().enumerate() {
        let colorized = match i {
            0 => String::new(),
            1..=5 => format!("{}", style(line).color256(208).bold()),
            6 => format!("{}", style(line).dim()),
            _ => colorize_structure(line),
        };
        if i == 0 {
            out.push('\n');
        } else {
            out.push_str(&colorized);
            out.push('\n');
        }
    }
    out
}

fn colorize_structure(line: &str) -> String {
    let mut out = String::with_capacity(line.len() * 4);
    for ch in line.chars() {
        out.push_str(&match ch {
            'O' => style(ch).red().bold().to_string(),
            'N' => style(ch).blue().bold().to_string(),
            'S' => style(ch).yellow().bold().to_string(),
            'H' => style(ch).color256(252).to_string(),
            'C' => style(ch).color256(244).to_string(),
            ' ' => " ".to_owned(),
            _ => style(ch).color256(241).to_string(),
        });
    }
    out
}

pub fn print_banner() {
    eprint!("{}", banner());
    eprintln!();
}

pub fn print_completion(label: &str, n: usize, elapsed: Duration, output: &Path) {
    eprintln!();
    eprintln!(
        "  {}  {} residues  {}  🎉",
        style(label).green().bold(),
        style(fmt(n)).bold(),
        style(dur(elapsed)).dim(),
    );
    eprintln!(
        "  {}  {}",
        style("wrote").green().bold(),
        style(output.display()).dim(),
    );
    eprintln!();
}

pub struct LoadSpinner {
    bar: Option<ProgressBar>,
    start: Instant,
}

impl LoadSpinner {
    pub fn new() -> Self {
        let b = ProgressBar::new_spinner();
        b.set_style(running_style());
        b.set_message("Reading and parameterizing...");
        b.enable_steady_tick(Duration::from_millis(80));
        Self {
            bar: Some(b),
            start: Instant::now(),
        }
    }

    pub fn done(mut self) {
        if let Some(b) = self.bar.take() {
            b.set_style(done_style());
            b.finish_with_message(done_line("IO/Parameterizing", "", self.start.elapsed()));
        }
    }
}

impl Drop for LoadSpinner {
    fn drop(&mut self) {
        if let Some(b) = self.bar.take() {
            b.finish_and_clear();
        }
    }
}

struct PhaseRow {
    bar: ProgressBar,
    name: &'static str,
    t: Mutex<Option<Instant>>,
}

impl PhaseRow {
    fn register(multi: &MultiProgress, name: &'static str) -> Self {
        let bar = multi.add(ProgressBar::new_spinner());
        bar.set_style(pending_style());
        bar.set_message(style(format!("{name:<22}")).dim().to_string());
        Self {
            bar,
            name,
            t: Mutex::new(None),
        }
    }

    fn start(&self, activity: &str) {
        *self.t.lock().unwrap() = Some(Instant::now());
        self.bar.set_message(format!(
            "{}  {}",
            style(format!("{:<22}", self.name)).bold(),
            style(activity).dim(),
        ));
        self.bar.set_style(running_style());
        self.bar.enable_steady_tick(Duration::from_millis(80));
    }

    fn finish(&self, stats: &str) {
        let elapsed = self
            .t
            .lock()
            .unwrap()
            .take()
            .map(|t| t.elapsed())
            .unwrap_or_default();
        self.bar.set_style(done_style());
        self.bar
            .finish_with_message(done_line(self.name, stats, elapsed));
    }
}

pub struct PhaseSpinner {
    sampling: PhaseRow,
    graph: PhaseRow,
    prune: PhaseRow,
    pair: PhaseRow,
    dee: PhaseRow,
    dp: PhaseRow,
    ctx: Mutex<Ctx>,
}

struct Ctx {
    total_conf: usize,
    n_graph_edges: usize,
    alive_conf: usize,
}

impl Default for PhaseSpinner {
    fn default() -> Self {
        let m = MultiProgress::new();
        Self {
            sampling: PhaseRow::register(&m, "Conformer Sampling"),
            graph: PhaseRow::register(&m, "Contact Graph"),
            prune: PhaseRow::register(&m, "Self-Energy Pruning"),
            pair: PhaseRow::register(&m, "Pair Energies"),
            dee: PhaseRow::register(&m, "Dead-End Elimination"),
            dp: PhaseRow::register(&m, "Tree-Decomp. DP"),
            ctx: Mutex::new(Ctx {
                total_conf: 0,
                n_graph_edges: 0,
                alive_conf: 0,
            }),
        }
    }
}

impl Progress for PhaseSpinner {
    fn sampling_begin(&self) {
        self.sampling.start("Sampling conformers...");
    }

    fn sampling_done(&self, total: usize, max: usize, min: usize) {
        self.ctx.lock().unwrap().total_conf = total;
        self.sampling
            .finish(&format!("{} conf · max {} · min {}", fmt(total), max, min));
    }

    fn graph_begin(&self) {
        self.graph.start("Building contact graph...");
    }

    fn graph_done(&self, n_edges: usize, degree_max: usize, n_isolated: usize) {
        self.ctx.lock().unwrap().n_graph_edges = n_edges;
        let stats = if n_isolated > 0 {
            format!(
                "{} edges · deg ≤{} · {} isolated",
                fmt(n_edges),
                degree_max,
                fmt(n_isolated)
            )
        } else {
            format!("{} edges · deg ≤{}", fmt(n_edges), degree_max)
        };
        self.graph.finish(&stats);
    }

    fn prune_begin(&self) {
        let n = self.ctx.lock().unwrap().total_conf;
        self.prune
            .start(&format!("Pruning {} conformers...", fmt(n)));
    }

    fn prune_done(&self, after: usize, trivial: usize) {
        let pct = {
            let mut c = self.ctx.lock().unwrap();
            c.alive_conf = after;
            if c.total_conf > 0 {
                100 - after * 100 / c.total_conf
            } else {
                0
            }
        };
        self.prune.finish(&format!(
            "{} alive · {}% pruned · {} resolved",
            fmt(after),
            pct,
            fmt(trivial)
        ));
    }

    fn pair_begin(&self) {
        let n = self.ctx.lock().unwrap().n_graph_edges;
        self.pair
            .start(&format!("Computing {} pair energies...", fmt(n)));
    }

    fn pair_done(&self, n_edges: usize, total_entries: usize) {
        self.pair.finish(&format!(
            "{} pairs · {} entries",
            fmt(n_edges),
            fmt(total_entries)
        ));
    }

    fn dee_begin(&self) {
        let n = self.ctx.lock().unwrap().alive_conf;
        self.dee
            .start(&format!("Eliminating dead-ends in {} conf...", fmt(n)));
    }

    fn dee_done(&self, eliminated: usize, trivial: usize) {
        self.dee.finish(&format!(
            "{} eliminated · {} resolved",
            fmt(eliminated),
            fmt(trivial)
        ));
    }

    fn dp_begin(&self) {
        self.dp.start("Solving global minimum energy...");
    }

    fn dp_done(&self) {
        self.dp
            .finish("Global Minimum Energy Conformation (GMEC) found");
    }
}

fn done_line(name: &str, stats: &str, elapsed: Duration) -> String {
    let time_s = dur(elapsed);
    let time_vis = console::measure_text_width(&time_s);
    let pre_pad = " ".repeat(8usize.saturating_sub(time_vis));

    let check = style("✓").green().bold();
    let name_s = style(format!("{name:<22}")).bold();
    let time_d = style(&time_s).dim();

    if stats.is_empty() {
        format!("  {check}  {name_s}  {pre_pad}{time_d}")
    } else {
        format!(
            "  {check}  {name_s}  {pre_pad}{time_d}  {}",
            style(stats).dim()
        )
    }
}

fn fmt(n: usize) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..10_000 => format!("{:.1}k", n as f64 / 1e3),
        10_000..1_000_000 => format!("{}k", n / 1_000),
        1_000_000..10_000_000 => format!("{:.1}m", n as f64 / 1e6),
        10_000_000..1_000_000_000 => format!("{}m", n / 1_000_000),
        _ => format!("{:.1}b", n as f64 / 1e9),
    }
}

fn dur(d: Duration) -> String {
    let us = d.as_micros();
    match us {
        0..1_000 => format!("{us}µs"),
        1_000..10_000 => format!("{:.2}ms", us as f64 / 1e3),
        10_000..100_000 => format!("{:.1}ms", us as f64 / 1e3),
        100_000..1_000_000 => format!("{:.0}ms", us as f64 / 1e3),
        1_000_000..10_000_000 => format!("{:.3}s", us as f64 / 1e6),
        _ => format!("{:.1}s", us as f64 / 1e6),
    }
}
