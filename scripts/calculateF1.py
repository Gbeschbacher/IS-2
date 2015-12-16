import csv
import glob

        
#Reading a file and returning its content
def readFile(filename, cols, header = False):
    content = []
    everything = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        if header:
            headers = reader.next()

        for row in reader:
            content.append(row)

    return content


# Main program
if __name__ == "__main__":

    for filename in glob.glob("./results/*.txt"):
        
        content = readFile(filename, [2,3], True)
            
        newContent = []
        
        for row in content :
            prec = float(row[2])
            rec = float(row[3])
            f1Score = 2* (prec*rec)/(prec+rec)
            row.append(f1Score)
            newContent.append(row)

        print newContent
        
        with open(filename + ".csv", "a") as myfile:
            for row in newContent :
                myfile.write(row[0] + "\t" + row[1] + "\t" + "%.2f" % float(row[2]) + "\t" + "%.2f" % float(row[3]) + "\t" + "%.2f" % float(row[5]) + "\t" + "%.2f" % float(row[4]) + "\n")
      