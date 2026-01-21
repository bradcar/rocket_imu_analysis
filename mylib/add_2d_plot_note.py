import matplotlib.pyplot as plt


def add_2d_plot_note(note_text, ax=None, x=0.65, y=0.10, fontsize=9, color="green"):
    """
    Adds a semi-transparent text box note to a plot. with smaller 9pt Italic font

    Parameters:
        note_text (str): The text to display.
        ax (matplotlib.axes.Axes, optional): Axis to place the note on. current axes is default.
        x (float): x-axis position
        y (float): y-axis position
        fontsize (int): Font size for note.
        color (str): The color of note.
    """
    if ax is None:
        ax = plt.gca()
    ax.text(
        x, y,
        note_text,
        transform=ax.transAxes,  # Use axis-relative coordinates
        fontsize=fontsize,
        fontstyle="italic",
        color=color,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="green")
    )
