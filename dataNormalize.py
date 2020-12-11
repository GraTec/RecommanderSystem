import re


def repfunc(matchObj):
    return matchObj.group().replace(',', '-')


with open('./BX-CSV-Dump/BX-Book-Ratings.csv', newline='', encoding='ISO-8859–1') as f:
    f = f.read()
    pattern = re.compile(r';".*,.*";')
    print(pattern.findall(f))
    f = re.sub(pattern, repfunc, f)
    pattern2 = re.compile(r';".*-.*";')
    print(pattern2.findall(f))
    out = open('./BX-CSV-Dump/dataset.csv', 'w')
    out.write(f)
    out.close()
