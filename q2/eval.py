import matplotlib.pyplot as plt


def load_results():
    with open("q2/results/results.txt") as f:
        lines = f.readlines()

    base = float(lines[0].split(":")[1])
    improved = float(lines[1].split(":")[1])

    return base, improved


def plot_results(base, improved):
    models = ["Baseline", "Improved"]
    acc = [base, improved]

    plt.bar(models, acc)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")

    plt.savefig("q2/results/comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    base, improved = load_results()

    print("Baseline:", base)
    print("Improved:", improved)

    plot_results(base, improved)