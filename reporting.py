import csv
import os
import matplotlib.pyplot as plt


def report_results(save_dir, results, save, thetas=None, losses=None):
    results_file = save_dir + "/" + save + ".csv"
    write_result(results_file, results)

    train_file = (
        save_dir
        + "/"
        + save
        + "_seed_{}".format(results["seed"])
        + "_method_{}".format(results["method"])
        + "_train.csv"
    )

    if results["method"] != "ewm":
        for i in range(len(thetas)):
            iteration_dic = {}
            iteration_dic["iteration"] = i
            iteration_dic["loss"] = losses[i]
            iteration_dic["params"] = list(thetas[i].flatten())
            write_result(train_file, iteration_dic)


#    plt.plot(list(range(len(losses))), losses)
#    plt.title("Learning at Equilibrium")
#    plt.xlabel("Iterations")
#    plt.ylabel("Loss")
#    plt.savefig("results/figures/{}.pdf".format(save))


def write_result(results_file, result):
    """Writes results to a csv file."""
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)
