print "<!DOCTYPE html><html><head><style>"
print "table.app {border:1px solid #d4d4d4;}"
print "#circle {border-radius:50% 50% 50% 50%;}</style></head><body background-color: transparent;><table class='app'>"


import sys
from itertools import groupby
path = sys.argv[1] + "/"
#path = '/Users/ninghuawang/Downloads/deep-learning-models-master/img/' + sys.argv[1] + "/"
for key, group in groupby(sys.stdin, lambda x: x.rstrip().split('|')[0]):
        print "<tr>"
        print "<td><img id='circle' src='" + path + key + "' style='height:150px; width:150px' /></td>"
        for item in group:
                item = item.rstrip().split('|')
		item[2] = str(int(float(item[2])*100))
                if key <> item[1]:
			print "<td><div class='image'><img id='circle' src='" + path + item[1] + \
				"' style='height:100px; width:100px' /><h4 align='center'>" + item[2] + "%</h4></div></td>"
        print "</tr>"


print "</table></body>"
