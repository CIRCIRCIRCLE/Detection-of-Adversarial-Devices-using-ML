import pandas as pd
import matplotlib.pyplot as plt


file_path = 'viz/test_res.xlsx'  
df = pd.read_excel(file_path)


grouped = df.groupby('Attack')
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
attack_methods = ['BitFlip', 'Lie', 'Min-Sum', 'IPM']
axs = axs.flatten()


colors = ['#992224', '#EF8B67', '#7895c1', '#80c5a2', '#a47dc0']


for ax, attack in zip(axs, attack_methods):
    attack_data = grouped.get_group(attack)
    for idx, (_, row) in enumerate(attack_data.iterrows()):
        method = row['Method']
        accuracies = list(map(float, row['Accuracies'].split(', ')))
        ax.plot(range(3, len(accuracies)), accuracies[3:], marker='o', label=method, color=colors[idx % len(colors)])  # 从 round 2 开始
    ax.set_xlabel('Round')
    ax.set_ylabel('Test Accuracy')
    ax.legend()
    ax.grid(True)



plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("test_accuracies_comparison.png")
plt.show()
