use anyhow::Result;
use clap::Parser;

mod urdf_logger;
use urdf_logger::parse_and_log_urdf_hierarchy;

#[derive(Parser, Debug)]
#[command(name = "rust_krecviz")]
struct Args {
    /// Path to the URDF file
    #[arg(long)]
    urdf: Option<String>,
}

fn main() -> Result<()> {
    // Show the parsed CLI args
    let args = dbg!(Args::parse());

    // 1) Start a Rerun viewer
    let rec = dbg!(rerun::RecordingStreamBuilder::new("rust_krecviz_hierarchy_example"))
        .spawn()?;

    // 2) If we have a URDF, parse & log it hierarchically
    if let Some(urdf_path) = &args.urdf {
        dbg!(urdf_path);
        parse_and_log_urdf_hierarchy(urdf_path, &rec)?;
    } else {
        dbg!("No URDF path provided, logging a fallback message.");
        rec.log("/urdf_info", &rerun::TextDocument::new("No URDF provided"))?;
    }

    // Sleep so we can see the result in the viewer
    std::thread::sleep(std::time::Duration::from_secs(5));
    Ok(())
}
