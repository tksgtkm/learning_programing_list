import math
import warnings

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class _Brewer:

    color_iter = None

    colors = [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ][::-1]

    which_colors = [
        [],
        [1],
        [1, 3],
        [0, 2, 4],
        [0, 2, 4, 6],
        [0, 2, 3, 5, 6],
        [0, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ]

    current_figure = None

    @classmethod
    def Colors(cls):
        return cls.colors
    
    @classmethod
    def ColorGenerator(cls, num):
        for i in cls.which_colors[num]:
            yield cls.colors[i]
        raise StopIteration("Run out of colors in _Brewer")
    
    @classmethod
    def InitIter(cls, num):
        cls.color_iter = cls.ColorGenerator(num)
        fig = plt.gcf()
        cls.current_figure = fig

    @classmethod
    def ClearIter(cls):
        cls.color_iter = None
        cls.current_figure = None

    @classmethod
    def GetIter(cls, num):
        fig = plt.gcf()
        if fig != cls.current_figure:
            cls.InitIter(num)
            cls.current_figure = fig

        if cls.color_iter is None:
            cls.InitIter(num)

        return cls.color_iter
    
def _UnderrideColor(options):
    if "color" in options:
        return options
    
    color_iter = _Brewer.GetIter(5)

    try:
        options["color"] = next(color_iter)
    except StopIteration:
        warnings.warn("Ran out of colors, Starting over")
        _Brewer.ClearIter()
        _UnderrideColor(options)

    return options

def PrePlot(num=None, rows=None, cols=None):

    if num:
        _Brewer.InitIter(num)

    if rows is None and cols is None:
        return
    
    if rows is not None and cols is None:
        cols = 1

    if cols is not None and rows is None:
        rows = 1

    size_map = {
        (1, 1): (8, 6),
        (1, 2): (12, 6),
        (1, 3): (12, 6),
        (1, 4): (12, 5),
        (1, 5): (12, 4),
        (2, 2): (10, 10),
        (2, 3): (16, 10),
        (3, 1): (8, 10),
        (4, 1): (8, 12),
    }

    if (rows, cols) in size_map:
        fig = plt.gcf()
        fig.set_size_inches(*size_map[rows, cols])

    if rows > 1 or cols > 1:
        ax = plt.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols
    else:
        ax = plt.gca()

    return ax

def SubPlot(plot_number, rows=None, cols=None, **options):
    rows = rows or SUBPLOT_ROWS
    cols = cols or SUBPLOT_COLS
    return plt.subplot(rows, cols, plot_number, **options)

def _Underride(d, **options):
    if d is None:
        d = {}
    
    for key, val in options.items():
        d.setdefault(key, val)

    return d

def Clf():
    global LOC
    LOC = None
    _Brewer.ClearIter()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

def Figure(**options):
    _Underride(options, figsize=(6, 8))
    plt.figure(**options)

def Plot(obj, ys=None, style="", **options):
    options = _UnderrideColor(options)
    label = getattr(obj, "label", "_nolegend_")
    options = _Underride(options, linewidth=3, alpha=0.7, label=label)

    xs = obj
    if ys is None:
        if hasattr(obj, "Render"):
            xs, ys = obj.Render()
        if isinstance(obj, pd.Series):
            ys = obj.values
            xs = obj.index

    if ys is None:
        plt.plot(xs, style, **options)
    else:
        plt.plot(xs, ys, style, **options)

def Vlines(xs, y1, y2, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.3)
    plt.vlines(xs, y1, y2, **options)

def Hlines(ys, x1, x2, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.hlines(ys, x1, x2, **options)

def axvline(x, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.axvline(x, **options)

def axhline(y, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.axhline(y, **options)

def tight_layout(**options):
    options = _Underride(
        options, wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1
    )
    plt.tight_layout()
    plt.subplots_adjust(**options)

def FillBetween(xs, y1, y2=None, where=None, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.5)
    plt.fill_between(xs, y1, y2, where, **options)

def Bar(xs, ys, **options):
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.6)
    plt.bar(xs, ys, **options)

def Scatter(xs, ys=None, **options):
    options = _Underride(options, color="blue", alpha=0.2, s=30, edgecolor="none")

    if ys is None and isinstance(xs, pd.Series):
        ys = xs.values
        xs = xs.index

    plt.scatter(xs, ys, **options)

def HexBin(xs, ys, **options):
    options = _Underride(options, cmap=matplotlib.cm.Blues)
    plt.hexbin(xs, ys, **options)

def Pdf(pdf, **options):
    low, high = options.pop("low", None), options.pop("high", None)
    n = options.pop("n", 101)
    xs, ps = pdf.Render(low=low, high=high, n=n)
    options = _Underride(options, label=pdf.label)
    Plot(xs, ps, **options)

def Pdfs(pdfs, **options):
    for pdf in pdfs:
        Pdf(pdf, **options)

def Hist(hist, **options):
    xs, ys = hist.Render()

    try:
        xs[0] - xs[0]
    except TypeError:
        labels = [str(x) for x in xs]
        xs = np.arange(len(xs))
        plt.xticks(xs + 0.5, labels)

    if "width" not in options:
        try:
            options["width"] = 0.9 * np.diff(xs).min()
        except TypeError:
            warnings.warn(
                "Hist: Can't compute bar width automatically"
                "Check for non-numeric types in Hist"
                "Or try providing width option"
            )

    options = _Underride(options, label=hist.label)
    options = _Underride(options, align="center")
    if options["align"] == "left":
        options["align"] = "edge"
    elif options["align"] == "right":
        options["align"] = "edge"
        options["width"] *= -1

    Bar(xs, ys, **options)

LEGEND = True
LOC = None

def Config(**options):
    names = [
        "title",
        "xlabel",
        "ylabel",
        "xscale",
        "yscale",
        "xticks",
        "yticks",
        "axis",
        "xlim",
        "ylim",
    ]

    for name in names:
        if name in options:
            getattr(plt, name)(options[name])

    global LEGEND

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    if LEGEND and len(labels) > 0:
        global LOC
        LOC = options.get("loc", LOC)
        frameon = options.get("frameon", True)

        try:
            plt.legend(loc=LOC, frameon=frameon)
        except UserWarning:
            pass
    
    val = options.get("xticklabels", None)
    if val is not None:
        if val == "invisible":
            ax = plt.gca()
            labels = ax.get_xticklabels()
            plt.setp(labels, visible=False)

    val = options.get("yticklabels", None)
    if val is not None:
        if val == "invisible":
            ax = plt.gca()
            labels = ax.get_yticklabels()
            plt.setp(labels, visible=False)

def Show(**options):
    clf = options.pop("clf", True)
    Config(**options)
    plt.show()
    if clf:
        Clf()

def Plotly(**options):
    clf = options.pop("clf", True)
    Config(**options)
    import plotly.plotly as plotly

    url = plotly.plot_mpl(plt.gcf())
    if clf:
        Clf()
    return url

def Save(root=None, formats=None, **options):
    clf = options.pop("clf", True)

    save_options = {}
    for option in ["bbox_inches", "pad_inches"]:
        if option in options:
            save_options[option] = options.pop(option)

    Config(**options)

    if formats is None:
        formats = ["pdf", "png"]

    try:
        formats.remove("plotly")
        Plotly(clf=False)
    except ValueError:
        pass

    if root:
        for fmt in formats:
            SaveFormat(root, fmt, **save_options)

    if clf:
        Clf()

def SaveFormat(root, fmt="eps", **options):
    _Underride(options, dpi=300)
    filename = "%s.%s" % (root, fmt)
    print("Writing", filename)
    plt.savefig(filename, format=fmt, **options)