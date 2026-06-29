import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lecture5_studen")

# Group by island and count penguins
penguin_count = df.groupby("island").size()

# Print result
print(penguin_count)

# Plot
plt.figure(figsize=(6,5))
penguin_count.plot(kind='bar')

plt.title("Number of Penguins by Island")
plt.xlabel("Island")
plt.ylabel("Number of Penguins")

plt.show()