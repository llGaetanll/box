use svg::node::element::ClipPath;
use svg::node::element::Definitions;
use svg::node::element::Group;
use svg::node::element::Line;
use svg::node::element::Polyline;
use svg::node::element::Rectangle;
use svg::node::element::Script;
use svg::node::element::Style;
use svg::node::element::Text;
use svg::Document;

use crate::BenchmarkData;

/// Base hues for benchmarks (HSL hue values 0-360).
const BASE_HUES: &[f64] = &[
    210.0, // blue
    120.0, // green
    30.0,  // orange
    280.0, // purple
    0.0,   // red
    180.0, // cyan
    330.0, // pink
    60.0,  // yellow
];

/// Chart dimensions and layout.
const CHART_WIDTH: f64 = 500.0;
const CHART_HEIGHT: f64 = 250.0;
const CHART_PADDING_LEFT: f64 = 40.0;
const CHART_PADDING_RIGHT: f64 = 10.0;
const CHART_PADDING_TOP: f64 = 40.0;
const CHART_PADDING_BOTTOM: f64 = 40.0;
const CHART_SPACING: f64 = 40.0;
const LEGEND_WIDTH: f64 = 70.0;
const LEGEND_LINE_HEIGHT: f64 = 20.0;

/// Lightness range for color shades (oldest to newest).
const LIGHTNESS_MIN: f64 = 35.0;
const LIGHTNESS_MAX: f64 = 70.0;
const SATURATION: f64 = 60.0;

/// Convert HSL to RGB hex string.
fn hsl_to_hex(h: f64, s: f64, l: f64) -> String {
    let s = s / 100.0;
    let l = l / 100.0;

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let r = ((r + m) * 255.0).round() as u8;
    let g = ((g + m) * 255.0).round() as u8;
    let b = ((b + m) * 255.0).round() as u8;

    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Generate colors for a series of runs, from lightest (oldest) to darkest (newest).
fn generate_shade_colors(base_hue: f64, count: usize) -> Vec<String> {
    if count == 0 {
        return vec![];
    }
    if count == 1 {
        return vec![hsl_to_hex(base_hue, SATURATION, LIGHTNESS_MIN)];
    }

    (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            let lightness = LIGHTNESS_MAX - t * (LIGHTNESS_MAX - LIGHTNESS_MIN);
            hsl_to_hex(base_hue, SATURATION, lightness)
        })
        .collect()
}

/// Generate an SVG group for a single benchmark chart.
fn generate_benchmark_chart(
    benchmark: &BenchmarkData,
    base_hue: f64,
    x_offset: f64,
    y_offset: f64,
    global_max_time: u64,
) -> Group {
    let mut group = Group::new();
    let chart_id = &benchmark.name;

    let runs = &benchmark.runs;
    if runs.is_empty() {
        return group;
    }

    let max_time = global_max_time;

    let max_frame = runs
        .iter()
        .flat_map(|r| r.frames.iter())
        .map(|f| f.frame)
        .max()
        .unwrap_or(1);

    let has_frames = runs.iter().any(|r| !r.frames.is_empty());
    if !has_frames {
        return group;
    }

    let plot_width = CHART_WIDTH - CHART_PADDING_LEFT - CHART_PADDING_RIGHT;
    let plot_height = CHART_HEIGHT - CHART_PADDING_TOP - CHART_PADDING_BOTTOM;
    let plot_x = x_offset + CHART_PADDING_LEFT;
    let plot_y = y_offset + CHART_PADDING_TOP;

    // Clip path for the plot area
    let clip_rect = Rectangle::new()
        .set("x", plot_x)
        .set("y", plot_y)
        .set("width", plot_width)
        .set("height", plot_height);
    let clip_path = ClipPath::new()
        .set("id", format!("clip-{}", chart_id))
        .add(clip_rect);
    let defs = Definitions::new().add(clip_path);
    group = group.add(defs);

    // Determine units and scale factor for display
    let (unit, scale) = if max_time >= 1_000_000 {
        ("s", 1_000_000.0)
    } else if max_time >= 1_000 {
        ("ms", 1_000.0)
    } else {
        ("μs", 1.0)
    };

    let max_time_scaled = max_time as f64 / scale;

    // Chart title
    let title = Text::new(benchmark.name.as_str())
        .set("id", format!("title-{}", chart_id))
        .set("data-base-name", benchmark.name.as_str())
        .set("data-unit", unit)
        .set("x", plot_x)
        .set("y", y_offset + 20.0)
        .set("font-family", "monospace")
        .set("font-size", 14)
        .set("font-weight", "bold");
    group = group.add(title);

    // Empty group for ticks - populated by JavaScript
    let ticks_group = Group::new()
        .set("id", format!("ticks-{}", chart_id))
        .set("data-plot-x", plot_x)
        .set("data-plot-y", plot_y)
        .set("data-plot-width", plot_width)
        .set("data-plot-height", plot_height);
    group = group.add(ticks_group);

    // X axis
    let x_axis = Line::new()
        .set("x1", plot_x)
        .set("y1", plot_y + plot_height)
        .set("x2", plot_x + plot_width)
        .set("y2", plot_y + plot_height)
        .set("stroke", "#aaa")
        .set("stroke-width", 1);
    group = group.add(x_axis);

    // Y axis
    let y_axis = Line::new()
        .set("x1", plot_x)
        .set("y1", plot_y)
        .set("x2", plot_x)
        .set("y2", plot_y + plot_height)
        .set("stroke", "#aaa")
        .set("stroke-width", 1);
    group = group.add(y_axis);

    // X axis label
    let x_label = Text::new("frame")
        .set("x", plot_x)
        .set("y", plot_y + plot_height + 32.0)
        .set("font-family", "monospace")
        .set("font-size", 11)
        .set("font-weight", "bold")
        .set("text-anchor", "start")
        .set("fill", "#000");
    group = group.add(x_label);

    // Y axis label
    let y_label = Text::new(unit)
        .set("x", x_offset + 8.0)
        .set("y", plot_y + plot_height)
        .set("font-family", "monospace")
        .set("font-size", 11)
        .set("font-weight", "bold")
        .set("text-anchor", "start")
        .set("fill", "#000")
        .set(
            "transform",
            format!("rotate(-90, {}, {})", x_offset + 8.0, plot_y + plot_height),
        );
    group = group.add(y_label);

    // Generate colors
    let colors = generate_shade_colors(base_hue, runs.len());

    // Data lines group with clip path
    let mut lines_group = Group::new()
        .set("id", format!("lines-{}", chart_id))
        .set("clip-path", format!("url(#clip-{})", chart_id))
        .set("data-plot-x", plot_x)
        .set("data-plot-y", plot_y)
        .set("data-plot-width", plot_width)
        .set("data-plot-height", plot_height)
        .set("data-min-frame", 0)
        .set("data-max-frame", max_frame)
        .set("data-min-time", 0.0)
        .set("data-max-time", max_time_scaled);

    // Draw data lines (oldest first so newest renders on top)
    for (i, run) in runs.iter().enumerate() {
        let color = &colors[i];
        let line_id = format!("line-{}-{}", chart_id, run.sha);

        if run.frames.is_empty() {
            continue;
        }

        let data_points: Vec<(u32, f64)> = run
            .frames
            .iter()
            .map(|frame| (frame.frame, frame.time_us as f64 / scale))
            .collect();
        let data_json = serde_json::to_string(&data_points).unwrap_or_default();

        let points: Vec<(f64, f64)> = run
            .frames
            .iter()
            .map(|frame| {
                let x = plot_x + (frame.frame as f64 / max_frame as f64) * plot_width;
                let y =
                    plot_y + plot_height - (frame.time_us as f64 / max_time as f64) * plot_height;
                (x, y)
            })
            .collect();

        let points_str: String = points
            .iter()
            .map(|(x, y)| format!("{:.1},{:.1}", x, y))
            .collect::<Vec<_>>()
            .join(" ");

        let polyline = Polyline::new()
            .set("id", line_id)
            .set("class", "data-line")
            .set("data-points", data_json)
            .set("points", points_str)
            .set("fill", "none")
            .set("stroke", color.as_str())
            .set("stroke-width", 1.5);
        lines_group = lines_group.add(polyline);
    }
    group = group.add(lines_group);

    // Selection rectangle (hidden by default)
    let selection_rect = Rectangle::new()
        .set("id", format!("selection-{}", chart_id))
        .set("class", "selection-rect")
        .set("x", 0)
        .set("y", 0)
        .set("width", 0)
        .set("height", 0)
        .set("fill", "rgba(180, 180, 180, 0.3)")
        .set("stroke", "none")
        .set("visibility", "hidden");
    group = group.add(selection_rect);

    // Cursor line
    let cursor_line = Line::new()
        .set("id", format!("cursor-line-{}", chart_id))
        .set("class", "cursor-line")
        .set("x1", 0)
        .set("y1", plot_y)
        .set("x2", 0)
        .set("y2", plot_y + plot_height)
        .set("stroke", "#ddd")
        .set("stroke-width", 1)
        .set("visibility", "hidden");
    group = group.add(cursor_line);

    // Cursor markers group
    let cursor_markers = Group::new()
        .set("id", format!("cursor-markers-{}", chart_id))
        .set("class", "cursor-markers");
    group = group.add(cursor_markers);

    // Cursor frame label
    let cursor_frame_label = Text::new("")
        .set("id", format!("cursor-frame-{}", chart_id))
        .set("class", "cursor-frame-label")
        .set("x", 0)
        .set("y", plot_y + plot_height + 15.0)
        .set("font-family", "monospace")
        .set("font-size", 11)
        .set("text-anchor", "middle")
        .set("visibility", "hidden");
    group = group.add(cursor_frame_label);

    // Cursor values group
    let cursor_values = Group::new()
        .set("id", format!("cursor-values-{}", chart_id))
        .set("class", "cursor-values");
    group = group.add(cursor_values);

    // Legend
    let legend_x = plot_x + plot_width + 15.0;
    let legend_y = plot_y;

    // Reset zoom button
    let reset_btn = Text::new("[reset zoom]")
        .set("id", format!("reset-{}", chart_id))
        .set("class", "reset-zoom")
        .set("data-chart-id", chart_id.as_str())
        .set("x", legend_x + 16.0)
        .set("y", legend_y - 10.0)
        .set("font-family", "monospace")
        .set("font-size", 12)
        .set("fill", "#666")
        .set("visibility", "hidden");
    group = group.add(reset_btn);

    for (i, run) in runs.iter().rev().enumerate() {
        let color_idx = runs.len() - 1 - i;
        let color = &colors[color_idx];
        let y = legend_y + (i as f64) * LEGEND_LINE_HEIGHT;
        let line_id = format!("line-{}-{}", chart_id, run.sha);
        let swatch_id = format!("swatch-{}-{}", chart_id, run.sha);

        // Color swatch
        let swatch = Rectangle::new()
            .set("id", swatch_id.clone())
            .set("x", legend_x)
            .set("y", y)
            .set("width", 12)
            .set("height", 12)
            .set("fill", color.as_str())
            .set("stroke", color.as_str())
            .set("stroke-width", 2);
        group = group.add(swatch);

        // SHA label (clickable)
        let label = Text::new(run.sha.as_str())
            .set("class", "legend-item")
            .set("data-line-id", line_id)
            .set("data-swatch-id", swatch_id)
            .set("data-color", color.as_str())
            .set("x", legend_x + 16.0)
            .set("y", y + 10.0)
            .set("font-family", "monospace")
            .set("font-size", 12);
        group = group.add(label);
    }

    // Invisible rectangle for mouse events
    let plot_area = Rectangle::new()
        .set("class", "plot-area")
        .set("data-chart-id", chart_id.as_str())
        .set("x", plot_x)
        .set("y", plot_y)
        .set("width", plot_width)
        .set("height", plot_height)
        .set("fill", "transparent");
    group = group.add(plot_area);

    group
}

/// CSS for interactive elements.
const CHART_CSS: &str = include_str!("../static/chart.css");

/// JavaScript for toggle, zoom, and tick functionality.
const CHART_JS: &str = include_str!("../static/chart.js");

/// Generate a complete SVG with charts for all benchmarks.
pub fn generate_svg(benchmarks: &[BenchmarkData]) -> String {
    let chart_total_width = CHART_WIDTH + LEGEND_WIDTH;
    let total_width = benchmarks.len() as f64 * (chart_total_width + CHART_SPACING);
    let total_height = CHART_HEIGHT;

    let global_max_time = benchmarks
        .iter()
        .flat_map(|b| b.runs.iter())
        .flat_map(|r| r.frames.iter())
        .map(|f| f.time_us)
        .max()
        .unwrap_or(1);

    let mut document =
        Document::new().set("viewBox", (0, 0, total_width as i32, total_height as i32));

    let style = Style::new(CHART_CSS);
    document = document.add(style);

    let background = Rectangle::new()
        .set("width", "100%")
        .set("height", "100%")
        .set("fill", "#fafafa");
    document = document.add(background);

    for (i, benchmark) in benchmarks.iter().enumerate() {
        let x_offset = i as f64 * (chart_total_width + CHART_SPACING);
        let y_offset = 0.0;
        let base_hue = BASE_HUES[i % BASE_HUES.len()];
        let chart_group =
            generate_benchmark_chart(benchmark, base_hue, x_offset, y_offset, global_max_time);
        document = document.add(chart_group);
    }

    let script = Script::new(CHART_JS);
    document = document.add(script);

    document.to_string()
}
