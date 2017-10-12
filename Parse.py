import matplotlib.pyplot as plt


timeframes = ["8h", "1d", "2d"]
for tf in timeframes:
	f = open("./output/" + tf + ".txt", "r")
	a = f.readlines()
	b = [x for x in a if x.startswith("Epoch")]

	epochs = [int(x[x.index("Epoch") + 6: x.index("Epoch") +  9]) for x in b]
	aucs   = [float(x[x.index("auc:") + 5: x.index("auc:") +  10]) for x in b]
	epochs_sparse = epochs[::5]
	aucs_sparse = []
	for i in range(0, len(aucs), 5):
	    aucs_sparse.append(max(aucs[i:i + 5]))

	plt.plot(epochs_sparse, aucs_sparse, "bo")
	plt.ylabel("AUC")
	plt.xlabel("Epoch")
	plt.title(tf + ": AUC vs Epoch")
	plt.axis([0, 300, 70, 90])
	plt.savefig("./output/plots/" + tf + ".pdf")
	plt.close()