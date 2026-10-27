import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/a3_results.csv")

# Chart 1: FedAvg accuracy by number of hospitals, holding E = 5
data1 = df[df["E"] == 5]

plt.figure(figsize=(9, 6))

for skew in data1["skew"].unique():
    subset = data1[data1["skew"] == skew].sort_values("n_clients")
    plt.plot(
        subset["n_clients"],
        subset["fedavg_acc"],
        marker="o",
        label=skew
    )

plt.xlabel("Number of hospitals")
plt.ylabel("FedAvg accuracy")
plt.title("FedAvg Accuracy by Consortium Size, E = 5")
plt.xticks([3, 5, 10])
plt.grid(True, alpha=0.3)
plt.legend(title="Skew")
plt.tight_layout()
plt.savefig("results/chart_fedavg_accuracy.png", dpi=300)
plt.close()

# Chart 2: Worst-hospital accuracy by skew, holding n_clients = 5
data2 = df[df["n_clients"] == 5]

pivot = data2.pivot(
    index="skew",
    columns="E",
    values="worst_hospital_acc"
)

skew_order = ["none", "label", "quantity", "feature", "concept", "realistic"]
pivot = pivot.reindex(skew_order)

ax = pivot.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_xlabel("Type of skew")
ax.set_ylabel("Worst-hospital accuracy")
ax.set_title("Worst-Hospital Accuracy by Skew, 5 Hospitals")
ax.grid(axis="y", alpha=0.3)
ax.legend(title="Local epochs, E")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/chart_worst_hospital_accuracy.png", dpi=300)
plt.close()

print("Both charts were created successfully.")
