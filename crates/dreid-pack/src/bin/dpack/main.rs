mod args;
mod config;
mod ui;

use std::io::{BufReader, BufWriter};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{CommandFactory, FromArgMatches};

fn main() {
    let matches = args::Cli::command().before_help(ui::banner()).get_matches();
    let cli = args::Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    if !cli.quiet {
        ui::print_banner();
    }

    if let Err(e) = run(&cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: &args::Cli) -> Result<()> {
    let (io, structure, charges, packing) = cli.command.common();

    let in_fmt = config::format_from_path(&io.input)?;
    let output = config::resolve_output(io);
    let out_fmt = config::format_from_path(&output)?;
    let scope = config::packing_scope(&cli.command)?;
    let read_cfg = config::read_config(structure, charges, packing, scope)?;
    let pack_cfg = config::pack_config(packing);

    let total = Instant::now();

    let in_file = std::fs::File::open(&io.input)
        .with_context(|| format!("cannot open '{}'", io.input.display()))?;
    let load_sp = (!cli.quiet).then(ui::LoadSpinner::new);

    let mut session = dreid_pack::io::read(BufReader::new(in_file), in_fmt, &read_cfg)
        .with_context(|| format!("failed to read '{}'", io.input.display()))?;

    if let Some(sp) = load_sp {
        sp.done();
    }

    let n_mobile = session.system.mobile.len();

    if cli.quiet {
        dreid_pack::pack::<()>(&mut session.system, &pack_cfg);
    } else {
        dreid_pack::pack::<ui::PhaseSpinner>(&mut session.system, &pack_cfg);
    }

    let out_file = std::fs::File::create(&output)
        .with_context(|| format!("cannot create '{}'", output.display()))?;
    dreid_pack::io::write(BufWriter::new(out_file), &session, out_fmt)
        .with_context(|| format!("failed to write '{}'", output.display()))?;

    if !cli.quiet {
        ui::print_completion(cli.command.label(), n_mobile, total.elapsed(), &output);
    }

    Ok(())
}
