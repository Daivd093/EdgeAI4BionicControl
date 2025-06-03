import time
with open("Test.txt") as file:
    for L in file:
        print(L.replace("\\n", "\n"), end="")
        time.sleep(0.5)
