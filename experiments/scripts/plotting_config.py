import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


class PlotConfig:
    """This class is used to make sure that all plots themes + fonts are
    unified across the project. To use the class do the following:

    Example:
    >> PlotConfig.setup() # set up the seaborn theme, font sizes, etc.
    >> fsize = PlotConfig.convert_width((5, 1), page_scale=1.0) # get the correct size figure
    >> fig, axs = plt.subplots(...) # make some plots
    >> PlotConfig.save_fig(fig, "my_plot") # save your plot in the unified format
    """

    # TODO: update to actual width of
    WIDTH_INCHES = 5.5
    MAJOR_FONT_SIZE = 8
    MINOR_FONT_SIZE = 6

    @classmethod
    def setup(cls):
        sns.set_theme(style="whitegrid")
        mpl.rc("text", usetex=True)
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.size"] = cls.MAJOR_FONT_SIZE
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["axes.labelsize"] = cls.MAJOR_FONT_SIZE
        plt.rcParams["xtick.labelsize"] = cls.MINOR_FONT_SIZE
        plt.rcParams["ytick.labelsize"] = cls.MINOR_FONT_SIZE
        plt.rcParams["axes.titlesize"] = cls.MAJOR_FONT_SIZE
        plt.rcParams["legend.fontsize"] = cls.MINOR_FONT_SIZE

    @classmethod
    def convert_width(
        cls, aspect_ratio: tuple[float, float], page_scale: float
    ) -> tuple[float, float]:
        """converts an arbitrary figure size into an appropriate figure size maintaining
        the aspect ration.

        Args:
            fsize (tuple[float, float]): original figure size
            page_scale(float): what percentage of the page should the figure take up

        Returns:
            tuple[float, float]: new figure size adjusted to appropriate width
        """
        rescale_width = cls.WIDTH_INCHES * page_scale
        width = aspect_ratio[0]
        new_fsize = tuple(size * rescale_width / width for size in aspect_ratio)
        assert len(new_fsize) == 2, "New fig size must be len(2)"
        return new_fsize

    @classmethod
    def save_fig(cls, fig, name: str):
        """Saves a figure with the appropriate size and resolution."""
        name = f"{name}.pdf"
        fig.savefig(name, format="pdf", dpi=1000, bbox_inches="tight")
