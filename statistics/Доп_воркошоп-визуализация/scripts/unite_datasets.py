import csv
from os import listdir

dest = open('data/united_wines.csv', 'w')
writer = csv.writer(dest, quoting=csv.QUOTE_MINIMAL)

is_first = True
for file_name in listdir('data'):
    if file_name == 'united_wines.csv': continue
    
    category = file_name[:-4]
    with open(f'data/{file_name}') as source:
        reader = csv.DictReader(source)
        if is_first:
            writer.writerow(reader.fieldnames + ['Category'])
            is_first = False
        for row in reader:
            writer.writerow(list(row.values()) + [category])

dest.close()