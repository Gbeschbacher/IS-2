import csv
import glob


# Main program
if __name__ == "__main__":

    for filename in glob.glob("./results/*.txt"):

        reader = csv.DictReader(open(filename, "r"), delimiter="\t")

        result = sorted(reader, key=lambda d: float(d["Artists"]))

        writer = csv.DictWriter(open(filename, "w"), fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(result)
