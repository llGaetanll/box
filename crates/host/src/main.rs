use std::error::Error;

mod gpu;
mod live_app;
mod window_surface;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    live_app::run_live()
}
