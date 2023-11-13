UNIQUE_LABEL_TEMPLATE = '=UNIQUE(FLATTEN({},{},{},{},{},{},{},{},{},{},{},{},{},{},{}))'
COUNT_WITH_BLANKS_TEMPLATE = '=IFS(J{} = "", {}, J{} <> "", {})'
COUNT_TEMPLATE = '=COUNTIF(FLATTEN({},{},{},{},{},{},{},{},{},{},{},{},{},{},{}),{})'
COUNT_BLANKS = '=COUNTBLANK(FLATTEN({},{},{},{},{},{},{},{},{},{},{},{},{},{},{}))'
NUM_IMAGES = 15
NEXT_IMAGE_ROW = 5


def get_functions(column, starting_row, count_starting_row):
    cells = []
    for i in range(NUM_IMAGES):
        cells.append('{}${}'.format(column, starting_row + i * NEXT_IMAGE_ROW))
    count_template = COUNT_TEMPLATE.format(*cells, 'J{}'.format(count_starting_row))
    count_blanks = COUNT_BLANKS.format(*cells)
    count_with_blanks = COUNT_WITH_BLANKS_TEMPLATE.format(count_starting_row, count_blanks, count_starting_row, count_template)
    return UNIQUE_LABEL_TEMPLATE.format(*cells), count_template, count_blanks
