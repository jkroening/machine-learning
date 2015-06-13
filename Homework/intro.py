import numpy

randmatrix = numpy.random.rand(4,3)
rownum = 0
for r  in randmatrix:
    rownum += 1
    colnum = 0
    for c in r:
        colnum += 1
        print 'Row %s, Column %s: %s' % (rownum, colnum, c)
