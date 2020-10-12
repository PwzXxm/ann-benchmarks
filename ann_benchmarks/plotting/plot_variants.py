from ann_benchmarks.plotting.metrics import all_metrics as metrics

all_plot_variants = {
    "recall/time": ("k-nn", "qps", True, True),
    "recall_0.8/time": ("k-nn_0.8", "qps", True, True),

    "recall/buildtime": ("k-nn", "build", False, True),
    "recall_0.8/buildtime": ("k-nn_0.8", "build", False, True),

    "recall/indexsize": ("k-nn", "indexsize", True, True),
    # "recall_0.8/indexsize": ("k-nn_0.8", "indexsize"),

    # "recall/distcomps": ("k-nn", "distcomps"),
    # "recall_0.8/distcomps": ("k-nn_0.8", "distcomps"),

    # "rel/time": ("rel", "qps", True, True),

    # "recall/candidates": ("k-nn", "candidates"),
    # "recall_0.8/candidates": ("k-nn_0.8", "candidates"),

    "recall/qpssize": ("k-nn", "queriessize", True, True),
    "recall_0.8/qpssize": ("k-nn_0.8", "queriessize", True, True),

    "eps/time": ("epsilon", "qps", True, True),
    "eps_0.8/time": ("epsilon_0.8", "qps", True, True),

    "largeeps/time": ("largeepsilon", "qps", True, True),
    "largeeps_0.8/time": ("largeepsilon_0.8", "qps", True, True)
}
