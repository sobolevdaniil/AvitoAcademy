import csv
import sys
 
file_name = sys.argv[1]

source = open(f'raw_data/{file_name}.csv')
dest = open(f'data/{file_name}.csv', 'w')

reader = csv.DictReader(source)
writer = csv.writer(dest, quoting=csv.QUOTE_MINIMAL)

writer.writerow(reader.fieldnames)

for row in reader:
    try:
        int(row['Year'])
    except:
        continue

    writer.writerow(row.values())

source.close()
dest.close()
