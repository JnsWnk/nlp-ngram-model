import subprocess
import os

folder = "./data"

# Calculate cross-entropy and perplexity for each training file with itself as test file
for file in os.listdir(folder):
    filepath = folder + "/" + file
    if file.endswith(".txt"):
        print("File: ", file)
        args = ["python", "main.py", "-N", "5", "--source", filepath]
        subprocess.run(args)

# Split Merkel into 90/10 train/test and evaluate
for i in range(2, 8):
    print("N: ", i)
    args = ["python", "main.py", "-N", str(i), "--split", "0.9", "--source", "./data/merkel-de.txt"]
    subprocess.run(args)

print("Train Merkel, test Lyrik: ")
args = ["python", "main.py", "-N", "5", "--source", "./data/merkel-de.txt", "--text", "./data/lyrik-de.txt"]
subprocess.run(args)

#Research
#Compare Dracula and Frankenstein to Romeo and Juliet
print("Trained on Romeo and Juliet, tested on Dracula: ")
args = ["python", "main.py", "-N", "5", "--source", "./books/romeo.txt", "--file", "./books/drac.txt"]
subprocess.run(args)


print("Trained on Romeo and Juliet, tested on Frankenstein: ")
args = ["python", "main.py", "-N", "5", "--source", "./books/romeo.txt", "--file", "./books/frank.txt"]
subprocess.run(args)