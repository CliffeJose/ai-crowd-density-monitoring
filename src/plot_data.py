import csv
import matplotlib.pyplot as plt

time = []
people = []

with open("../Data/crowd_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        time.append(row[0])
        people.append(int(row[1]))

plt.figure()
plt.plot(time, people)
plt.xlabel("Time")
plt.ylabel("People Count")
plt.title("Crowd Density Over Time")
plt.xticks(rotation=45)
plt.tight_layout()

# Save graph
plt.savefig("../outputs/crowd_graph.png")

# Show graph
plt.show()
