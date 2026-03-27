# ============================================================
# Training Study Timeline - R Visualization
# Uses the vistime package to render a ggplot2-based timeline
# ============================================================

# Load required libraries
library(vistime)
library(ggplot2)

# ============================================================
# Create the timeline data frame
# Single row timeline with training period and retention checks
# ============================================================

timeline_data <- data.frame(
  event = c(
    "Voluntary Training",    # Training period (bar)
    "3-Month Retention",     # Retention check (dot)
    "Survey Measure +\n 6-Month Retention",     # Retention check (dot)
    "9-Month Retention",     # Retention check (dot)
    "12-Month Retention"     # Retention check (dot)
  ),
  start = c(
    "2026-01-01",  # Training starts Jan 1
    "2026-04-01",  # 3-month check
    "2026-06-01",  # 6-month check
    "2026-09-01",  # 9-month check
    "2026-12-01"   # 12-month check
  ),
  end = c(
    "2026-03-31",  # Training ends Mar 31
    "2026-04-01",  # Point-in-time (same day)
    "2026-06-01",  # Point-in-time (same day)
    "2026-09-01",  # Point-in-time (same day)
    "2026-12-01"   # Point-in-time (same day)
  ),
  group = "Timeline",  # Single group for single row
  color = c(
    "#AE94E5",                    # Voluntary Training (bar)
    rep("#6B8BFF", 4)              # Retention circles (second color)
  ),
  stringsAsFactors = FALSE
)

# ============================================================
# Render the timeline using gg_vistime()
# Customize with ggplot2 theme to remove grid lines
# Add vertical dotted line for L&D Review in July
# Fix x-axis to show only 2026 (January through December)
# ============================================================

p <- gg_vistime(
  timeline_data,
  col.event  = "event",
  col.start  = "start",
  col.end    = "end",
  col.group  = "group",
  title      = "Study Timeline",
  background_lines = 0
) +

  scale_x_date(
    limits = c(as.Date("2026-01-01"), as.Date("2026-12-31")),
    date_breaks = "1 month",
    date_labels = "%b"
  ) +

  theme(
    panel.grid.major = element_blank(),   
    panel.grid.minor = element_blank(),   
    panel.background = element_blank(),  
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank(),
    axis.line.y = element_blank(),       # Remove y-axis line
    aspect.ratio = 0.25                
  ) 


print(p)

#save
ggsave("./diagrams/timeline.png", p, width = 12, height = 8, dpi = 300)