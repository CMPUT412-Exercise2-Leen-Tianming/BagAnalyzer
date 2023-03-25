import os


IM_DIR = '../plots'


def main():
    i = 0
    for path in os.listdir(IM_DIR):
        if path[-4:] != '.png':
            continue
        substrs = path.split('_')
        time = int(substrs[1])
        if time <= 370:
            label = 3
        elif time <= 590:
            label = 6
        elif time <= 735:
            label = 0
        elif time <= 850:
            label = 2
        elif time <= 1055:
            label = 4
        elif time <= 1230:
            label = 7
        elif time <= 1355:
            label = 3
        elif time <= 1495:
            label = 9
        elif time <= 1685:
            label = 7
        elif time <= 1810:
            label = 4
        else:
            label = 9
        newname = f'{label}_{i}.png'
        os.rename(IM_DIR + '/' + path, IM_DIR + '/' + newname)
        strs = newname.split('_')
        print(strs[0], strs[1][:-4])
        i += 1


if __name__ == '__main__':
    main()
