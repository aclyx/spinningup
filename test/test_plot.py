import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from spinup.utils.plot import plot_data


def test_plot_data_with_modern_seaborn():
    data = pd.DataFrame(
        {
            "TotalEnvInteracts": [100, 200, 100, 200],
            "Performance": [1.0, 2.0, 1.5, 2.5],
            "Unit": [0, 0, 1, 1],
            "Condition1": ["ppo", "ppo", "ppo", "ppo"],
        }
    )

    plot_data(
        data,
        xaxis="TotalEnvInteracts",
        value="Performance",
        condition="Condition1",
    )

    assert plt.gca().lines
    plt.close("all")
