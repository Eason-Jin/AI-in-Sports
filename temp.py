import csv

file_path = 'NFL TRACKING/WK3_Ram_VS_49ers.csv' # yellow = away(ram), red = home(49ers)

values = set()

try:
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)

        for row in csv_reader:
            val = row[header.index('event')]
            if val != '':
                values.add(val)
    print(sorted(values))
except Exception as e:
    print(e)