import time
import h5py
import collections
import concurrent.futures
import threading
import matplotlib as mpl
mpl.use('Agg')  # noqa
import argparse
import os
import json
import pickle
import yaml
import numpy
import hashlib
from jinja2 import Environment, FileSystemLoader

from ann_benchmarks import results
from ann_benchmarks.algorithms.definitions import get_algorithm_name
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.plot_variants import (all_plot_variants
                                                   as plot_variants)
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (get_plot_label, compute_metrics,
                                           compute_all_metrics,
                                           create_pointset,
                                           create_linestyles)
import plot

colors = [
    "rgba(166,206,227,1)",
    "rgba(31,120,180,1)",
    "rgba(178,223,138,1)",
    "rgba(51,160,44,1)",
    "rgba(251,154,153,1)",
    "rgba(227,26,28,1)",
    "rgba(253,191,111,1)",
    "rgba(255,127,0,1)",
    "rgba(202,178,214,1)"
]

point_styles = {
    "o": "circle",
    "<": "triangle",
    "*": "star",
    "x": "cross",
    "+": "rect",
}


def convert_color(color):
    r, g, b, a = color
    return "rgba(%(r)d, %(g)d, %(b)d, %(a)d)" % {
        "r": r * 255, "g": g * 255, "b": b * 255, "a": a}


def convert_linestyle(ls):
    new_ls = {}
    for algo in ls.keys():
        algostyle = ls[algo]
        new_ls[algo] = (convert_color(algostyle[0]),
                        convert_color(algostyle[1]),
                        algostyle[2], point_styles[algostyle[3]])
    return new_ls


def get_run_desc(properties):
    # import pdb;pdb.set_trace()
    return "%(dataset)s_%(count)d_%(distance)s" % properties


def get_dataset_from_desc(desc):
    return desc.split("_")[0]


def get_count_from_desc(desc):
    return desc.split("_")[1]


def get_distance_from_desc(desc):
    return desc.split("_")[2]


def get_dataset_label(desc):
    return "{} (k = {})".format(get_dataset_from_desc(desc),
                                get_count_from_desc(desc))


def directory_path(s):
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError("'%s' is not a directory" % s)
    return s + "/"


def prepare_data(data, xn, yn):
    """Change format from (algo, instance, dict) to (algo, instance, x, y)."""
    res = []
    for algo, algo_name, result in data:
        if xn == "rel" and result[xn] == float("inf"):
            continue

        res.append((algo, algo_name, result[xn], result[yn]))
    return res


parser = argparse.ArgumentParser()
parser.add_argument(
    '--plottype',
    help='Generate only the plots specified',
    nargs='*',
    choices=plot_variants.keys(),
    default=plot_variants.keys())
parser.add_argument(
    '--outputdir',
    help='Select output directory',
    default='.',
    type=directory_path,
    action='store')
parser.add_argument(
    '--latex',
    help='generates latex code for each plot',
    action='store_true')
parser.add_argument(
    '--scatter',
    help='create scatterplot for data',
    action='store_true')
parser.add_argument(
    '--recompute',
    help='Clears the cache and recomputes the metrics',
    action='store_true')
args = parser.parse_args()


def get_lines(all_data, xn, yn, render_all_points):
    """ For each algorithm run on a dataset, obtain its performance
    curve coords."""
    plot_data = []
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
        xs, ys, ls, axs, ays, als = \
            create_pointset(prepare_data(all_data[algo], xn, yn), xn, yn)
        if render_all_points:
            xs, ys, ls = axs, ays, als
        plot_data.append({"name": algo, "coords": zip(xs, ys), "labels": ls,
                          "scatter": render_all_points})
    return plot_data


def create_plot(all_data, xn, yn, linestyle, j2_env, additional_label="",
                plottype="line"):
    xm, ym = (metrics[xn], metrics[yn])
    render_all_points = plottype == "bubble"
    plot_data = get_lines(all_data, xn, yn, render_all_points)
    latex_code = j2_env.get_template("latex.template").\
        render(plot_data=plot_data, caption=get_plot_label(xm, ym),
               xlabel=xm["description"], ylabel=ym["description"])
    plot_data = get_lines(all_data, xn, yn, render_all_points)
    button_label = hashlib.sha224((get_plot_label(xm, ym) + additional_label)
                                  .encode("utf-8")).hexdigest()
    return j2_env.get_template("chartjs.template").\
        render(args=args, latex_code=latex_code, button_label=button_label,
               data_points=plot_data,
               xlabel=xm["description"], ylabel=ym["description"],
               plottype=plottype, plot_label=get_plot_label(xm, ym),
               label=additional_label, linestyle=linestyle,
               render_all_points=render_all_points)


def build_detail_site(data, label_func, j2_env, linestyles, batch=False):
    for (name, runs) in data.items():
        print("Building '%s'" % name)
        all_runs = runs.keys()
        label = label_func(name)
        data = {"normal": [], "scatter": [], "if_plot": [], "if_scatter": []}

        for plottype in args.plottype:
            xn, yn, line, scatter = plot_variants[plottype]
            if line:
                data["if_plot"].append("true")
            else:
                data["if_plot"].append("false")

            data["normal"].append(create_plot(
                runs, xn, yn, convert_linestyle(linestyles), j2_env))
            if args.scatter and scatter:
                data["if_scatter"].append("true")
            else:
                data["if_scatter"].append("false")

            if args.scatter:
                data["scatter"].append(
                    create_plot(runs, xn, yn, convert_linestyle(linestyles),
                                j2_env, "Scatterplot ", "bubble"))

        # create png plot for summary page
        data_for_plot = {}
        for k in runs.keys():
            data_for_plot[k] = prepare_data(runs[k], 'k-nn', 'qps')
        plot.create_plot(
            data_for_plot, False,
            False, True, 'k-nn', 'qps',
            args.outputdir + get_algorithm_name(name, batch) + ".png",
            linestyles, batch)
        output_path = "".join([args.outputdir,
                               get_algorithm_name(name, batch),
                               ".html"])
        with open(output_path, "w") as text_file:
            text_file.write(j2_env.get_template("detail_page.html").
                            render(title=label, plot_data=data,
                                   args=args, batch=batch))


def build_index_site(datasets, algorithms, j2_env, file_name):
    dataset_data = {'batch': [], 'non-batch': []}
    for mode in ['batch', 'non-batch']:
        distance_measures = sorted(
            set([get_distance_from_desc(e) for e in datasets[mode].keys()]))
        sorted_datasets = sorted(
            set([get_dataset_from_desc(e) for e in datasets[mode].keys()]))

        for dm in distance_measures:
            d = {"name": dm.capitalize(), "entries": []}
            for ds in sorted_datasets:
                matching_datasets = [e for e in datasets[mode].keys()
                                     if get_dataset_from_desc(e) == ds and  # noqa
                                     get_distance_from_desc(e) == dm]
                sorted_matches = sorted(
                    matching_datasets,
                    key=lambda e: int(get_count_from_desc(e)))
                for idd in sorted_matches:
                    d["entries"].append(
                        {"name": idd, "desc": get_dataset_label(idd)})
            dataset_data[mode].append(d)

    with open(args.outputdir + "index.html", "w") as text_file:
        text_file.write(j2_env.get_template("summary.html").
                        render(title="ANN-Benchmarks",
                               dataset_with_distances=dataset_data,
                               algorithms=algorithms,
                               label_func=get_algorithm_name))


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("----run {} cost {}s".format(func.__name__, round(end - start, 2)))
        return result
    return wrapper


cached_true_neighbors = dict()


def process(ds):
    c, properties, root, fn = ds
    f = h5py.File(os.path.join(root, fn), 'r+')
    print("--------process pos: %s" % c)
    sdn = get_run_desc(properties)
    if sdn not in cached_true_neighbors:
        dataset = get_dataset(properties["dataset"])
        cached_true_neighbors[sdn] = list(dataset["neighbors"])
    ms = compute_all_metrics(
        cached_true_neighbors[sdn], f, properties, args.recompute)
    idx = "non-batch"
    if properties["batch_mode"]:
        idx = "batch"
    f.close()

    return [c, sdn, properties, idx, ms]


@time_wrapper
def load_all_results():
    """Read all result files and compute all metrics"""
    all_runs_by_dataset = {'batch': {}, 'non-batch': {}}
    all_runs_by_algorithm = {'batch': {}, 'non-batch': {}}

    # {position: [sdn, properties, idx, ms]}
    cache = collections.OrderedDict()

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future_results = {executor.submit(process, (c, p, r, f))
                          for c, (p, r, f) in enumerate(results.load_all_results_v2())}
        for future in concurrent.futures.as_completed(future_results):
            d = future.result()
            cache[d[0]] = d[1:]

    for pos, v in sorted(cache.items()):
        print("pos: %s" % pos)
        sdn = v[0]
        properties = v[1]
        idx = v[2]
        ms = v[3]
        algo = properties["algo"]
        algo_ds = get_dataset_label(sdn)
        all_runs_by_algorithm[idx].setdefault(
            algo, {}).setdefault(algo_ds, []).append(ms)
        all_runs_by_dataset[idx].setdefault(
            sdn, {}).setdefault(algo, []).append(ms)

    return (all_runs_by_dataset, all_runs_by_algorithm)


j2_env = Environment(loader=FileSystemLoader("./templates/"), trim_blocks=True)
j2_env.globals.update(zip=zip, len=len)
runs_by_ds, runs_by_algo = load_all_results()
dataset_names = [get_dataset_label(x) for x in list(
    runs_by_ds['batch'].keys()) + list(runs_by_ds['non-batch'].keys())]
algorithm_names = list(runs_by_algo['batch'].keys(
)) + list(runs_by_algo['non-batch'].keys())

linestyles = {**create_linestyles(dataset_names),
              **create_linestyles(algorithm_names)}

build_detail_site(
    runs_by_ds['non-batch'],
    lambda label: get_dataset_label(label), j2_env, linestyles, False)

build_detail_site(
    runs_by_ds['batch'],
    lambda label: get_dataset_label(label), j2_env, linestyles, True)

build_detail_site(
    runs_by_algo['non-batch'],
    lambda x: x, j2_env, linestyles, False)

build_detail_site(
    runs_by_algo['batch'], lambda x: x, j2_env, linestyles, True)

build_index_site(runs_by_ds, runs_by_algo, j2_env, "index.html")
