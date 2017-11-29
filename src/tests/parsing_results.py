results = []
with open('results.txt') as f:
    lines = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
for line in lines:
    numbers = line.strip()
    res = 0
    if numbers.__len__() != 6:
        results.append('brak danych')
        continue
    if numbers[0] == '1':
        res += 1
    if numbers[1] == '0':
        res += 1
    if numbers[2] == '1':
        res += 1
    if numbers[3] == '0':
        res += 1
    if numbers[4] == '1':
        res += 1
    if numbers[5] == '0':
        res += 1

    results.append(res)

thefile = open('parsed_resultes.txt', 'w')
for item in results:
    thefile.write("%s\n" % item)

content = [x.strip() for x in lines]
