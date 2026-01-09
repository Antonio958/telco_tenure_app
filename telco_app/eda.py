
import matplotlib.pyplot as plt

class EDAAnalyzer:
    def resumen_tenure_por(self, df, col):
        return (
            df.groupby(col)["tenure"]
              .agg(["count","mean","median","min","max"])
              .sort_values("mean", ascending=False)
        )

    def boxplot_tenure(self, df, col, title, rotation=0):
        plt.figure()
        df.boxplot(column="tenure", by=col)
        plt.title(title)
        plt.suptitle("")
        plt.xlabel(col)
        plt.ylabel("tenure")
        if rotation:
            plt.xticks(rotation=rotation, ha="right")
        plt.show()
