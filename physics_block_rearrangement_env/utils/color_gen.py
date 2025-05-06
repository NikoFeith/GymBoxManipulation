import colorsys

def generate_equally_spaced_colors(n, light=False, alpha=1.0):
    """Generate `n` visually distinct RGBA colors by evenly spacing hues."""
    colors = []
    for i in range(n):
        hue = i / n  # even hue spacing
        sat = 0.6 if light else 1.0
        val = 1.0 if light else 0.8
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append([r, g, b, alpha])
    return colors
