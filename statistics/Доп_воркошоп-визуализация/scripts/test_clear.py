import csv
import sys
 
file_name = sys.argv[1]
source = open(f'data/{file_name}.csv')

reader = csv.DictReader(source)

for i, row in enumerate(reader):
    try:
        int(row['Year'])
        float(row['Rating'])
        int(row['NumberOfRatings'])
        float(row['Price'])
    except:
        print(f"Строка {i}")
        raise

source.close()