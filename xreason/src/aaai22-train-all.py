#!/usr/bin/env python
##
## train-all.py
##
##  Created on: Aug 27, 2021
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
import getopt
import os
import sys


#
#==============================================================================
def parse_options():
    """
        Standard options handling.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:hn:v',
                ['depth=', 'help', 'num=', 'verbose'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    depth = 5
    num = 50
    verbose = False

    for opt, arg in opts:
        if opt in ('-d', '--depth'):
            depth = str(arg)
            if depth == 'none':
                depth = -1
            else:
                depth = int(depth)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-n', '--num'):
            num = int(arg)
        elif opt in ('-v', '--verbose'):
            verbose = True
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return depth, num, verbose, args


#
#==============================================================================
def usage():
    """
        Prints usage message.
        """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] datasets-list')
    print('Options:')
    print('        -d, --depth=<int>    Tree depth')
    print('                             Available values: [1 .. INT_MAX], none (default = 5)')
    print('        -h, --help           Show this message')
    print('        -n, --num=<int>      Number of trees per class')
    print('                             Available values: [1 .. INT_MAX] (default = 50)')
    print('        -v, --verbose        Be verbose')


#
#==============================================================================
if __name__ == '__main__':
    depth, num, verbose, files = parse_options()

    if files:
        datasets = files[0]
    else:
        datasets = 'datasets.list'

    with open(datasets, 'r') as fp:
        datasets = [line.strip() for line in fp.readlines() if line]

    print(f'training parameters: {num} trees per class, each of depth {"adaptive" if depth == -1 else depth}\n')

    # # training all XGBoost models
    # for data in datasets:
    #     if depth != -1:
    #         print(data)
    #         os.system(f'./xreason.py -t -n {num} -d {depth} {data}')
    #     else:
    #         data, adepth = data.split()
    #         print(data)
    #         os.system(f'./xreason.py -t -n {num} -d {adepth} {data}')


# param_dict = {
#     "arrowhead_0_vs_1.csv": (7, 100),
#     "arrowhead_0_vs_2.csv": (7, 100),
#     "arrowhead_1_vs_2.csv": (7, 100),
#     "australian.csv": (8, 50),
#     "balance_0_vs_1.csv": (6, 100),
#     "balance_0_vs_2.csv": (6, 100),
#     "balance_1_vs_2.csv": (6, 100),    
#     "bank.csv": (4, 200),    
#     "biodegradation.csv": (3, 100),    
#     "breast-tumor.csv": (3, 100),    
#     "bupa.csv": (6, 200),    
#     "cleveland.csv": (3, 100),    
#     "cnae.csv": (3, 100),    
#     "compas.csv": (6, 100),    
#     "contraceptive.csv": (3, 50),   
#     "divorce.csv": (4, 100),    
#     "german.csv": (5, 100),   
# }
    param_dict = {
    "arrowhead_0_vs_1.csv": (8, 150),
    "arrowhead_0_vs_2.csv": (8, 150),
    "arrowhead_1_vs_2.csv": (8, 150),
    "australian.csv": (8, 100),
    "balance_0_vs_1.csv": (6, 100),
    "balance_0_vs_2.csv": (6, 100),
    "balance_1_vs_2.csv": (6, 100),    
    "bank.csv": (4, 200),    
    "biodegradation.csv": (9, 200),    
    "breast-tumor.csv": (7, 200),    
    "bupa.csv": (6, 200),    
    "cleveland.csv": (7, 150),    
    "cnae.csv": (3, 100),    
    "compas.csv": (6, 100),    
    "contraceptive.csv": (6, 150),   
    "divorce.csv": (7, 150),    
    "german.csv": (5, 100),
    }

for data in datasets:
    adepth, anum = param_dict.get(os.path.basename(data), (depth, num))
    os.system(f'./xreason.py -t -n {anum} -d {adepth} {data}')

