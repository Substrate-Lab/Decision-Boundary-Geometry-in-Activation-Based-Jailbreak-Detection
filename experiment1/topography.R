#!/usr/bin/env Rscript
# Smooth 3D terrain of the power-diagram decision geometry, one figure per layer.
#
# Reads the grid + point CSVs written by export_topography.py and draws a shaded persp
# surface: Benign is flat blue ground, Refusal rises as a green mountain and Jailbreak as a
# red one, height = geometric margin. Facet colour blends from blue at the base to the
# winning class colour at the peak, so low ground stays blue and only cores light up.

suppressWarnings(suppressMessages({
	library(grDevices)
}))

# Resolve this script's own directory whether sourced or run via Rscript.
script_dir <- function() {
	full <- commandArgs(FALSE)
	file_arg <- sub("^--file=", "", full[grep("^--file=", full)])
	if (length(file_arg) > 0) return(dirname(normalizePath(file_arg)))
	ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
	if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
	getwd()
}
here <- script_dir()
args <- commandArgs(trailingOnly = TRUE)
export_dir <- file.path(here, "data", "topography")
figures_dir <- file.path(here, "figures")
if (!dir.exists(figures_dir)) dir.create(figures_dir, recursive = TRUE)

class_colors <- c(Refusal = "#1ecb96", Jailbreak = "#e5484d", Benign = "#0d2b8c")
base_blue <- "#0d2b8c"

# Blend two hex colours by a fraction in 0..1.
blend <- function(a, b, fraction) {
	ca <- col2rgb(a) / 255
	cb <- col2rgb(b) / 255
	mixed <- (1 - fraction) * ca + fraction * cb
	rgb(mixed[1], mixed[2], mixed[3])
}

# Draw one layer's terrain to a PNG.
render_layer <- function(layer) {
	grid <- read.csv(file.path(export_dir, sprintf("grid_layer%d.csv", layer)), stringsAsFactors = FALSE)
	points <- read.csv(file.path(export_dir, sprintf("points_layer%d.csv", layer)), stringsAsFactors = FALSE)

	ux <- sort(unique(grid$gx))
	uy <- sort(unique(grid$gy))
	z <- matrix(NA_real_, nrow = length(ux), ncol = length(uy))
	winner <- matrix("Benign", nrow = length(ux), ncol = length(uy))
	xi <- match(grid$gx, ux)
	yi <- match(grid$gy, uy)
	for (k in seq_len(nrow(grid))) {
		z[xi[k], yi[k]] <- grid$height[k]
		winner[xi[k], yi[k]] <- grid$winner[k]
	}

	# Facet height and winning class come from the four corners of each cell.
	nr <- nrow(z); nc <- ncol(z)
	facet_h <- (z[-nr, -nc] + z[-1, -nc] + z[-nr, -1] + z[-1, -1]) / 4
	facet_win <- winner[-nr, -nc]
	facet_color <- matrix(base_blue, nrow = nr - 1, ncol = nc - 1)
	for (i in seq_len(nr - 1)) {
		for (j in seq_len(nc - 1)) {
			cls <- facet_win[i, j]
			if (cls == "Benign" || facet_h[i, j] <= 0) {
				facet_color[i, j] <- base_blue
			} else {
				facet_color[i, j] <- blend(base_blue, class_colors[[cls]], min(1, facet_h[i, j]))
			}
		}
	}

	out_path <- file.path(figures_dir, sprintf("power_terrain_layer%d.png", layer))
	png(out_path, width = 1500, height = 1250, res = 150)
	par(mar = c(1, 1, 3, 1), bg = "white")
	persp(
		x = ux, y = uy, z = z,
		theta = -40, phi = 40, expand = 0.55,
		col = facet_color, border = NA, shade = 0.35, ltheta = -30, lphi = 45,
		xlab = "LDA-1", ylab = "LDA-2", zlab = "geometric margin",
		ticktype = "detailed", nticks = 4, zlim = c(0, 1.05),
		main = sprintf("Layer %d: harmful decision geometry (Benign = blue ground)", layer)
	)
	legend("topleft", legend = c("Refusal (green peak)", "Jailbreak (red peak)", "Benign (blue ground)"),
		fill = c(class_colors[["Refusal"]], class_colors[["Jailbreak"]], base_blue), border = NA, bty = "n", cex = 0.9)
	mtext("height = geometric margin; peaks rise only where a harmful class wins by a wide margin", side = 1, line = -1, cex = 0.8)
	dev.off()
	cat(sprintf("wrote %s\n", out_path))
}

# Layers come from CLI args, else every grid CSV that was exported.
if (length(args) > 0) {
	layers <- as.integer(args)
} else {
	files <- list.files(export_dir, pattern = "^grid_layer[0-9]+\\.csv$")
	layers <- sort(as.integer(sub("^grid_layer([0-9]+)\\.csv$", "\\1", files)))
}
for (layer in layers) render_layer(layer)
