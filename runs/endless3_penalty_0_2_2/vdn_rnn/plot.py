import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./eval_log.csv")
plt.plot(df["episode"], df["eval_metric"])
plt.xlabel("Training Episode")
plt.ylabel("Eval Average Return")  # 或者 Average Steps
plt.grid(True)
plt.show()