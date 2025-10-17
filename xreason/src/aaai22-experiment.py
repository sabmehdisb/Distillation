#!/usr/bin/env python
##
## experiment.py
##
##  Created on: Aug 27, 2021
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
from options import Options
import os
import random
import shutil
import subprocess
import sys
from xgbooster import XGBooster
import resource
import statistics


#
#==============================================================================
def parse_options():
    """
        Standard options handling.
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:hi:n:r:v',
                ['depth=', 'help', 'inst=',  'num=', 'relax=', 'verbose'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    depth = 5
    inst = 0.3
    num = 50
    relax = 0
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
        elif opt in ('-i', '--inst'):
            inst = float(arg)
        elif opt in ('-n', '--num'):
            num = int(arg)
        elif opt in ('-r', '--relax'):
            relax = int(arg)
        elif opt in ('-v', '--verbose'):
            verbose = True
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return depth, num, inst, relax, verbose, args


#
#==============================================================================
def usage():
    """
        Prints usage message.
        """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] datasets-list')
    print('Options:')
    print('        -d, --depth=<int>         Tree depth')
    print('                                  Available values: [1 .. INT_MAX], none (default = 5)')
    print('        -h, --help                Show this message')
    print('        -i, --inst=<float,int>    Fraction or number of instances to explain')
    print('                                  Available values: (0 .. 1] or [1 .. INT_MAX] (default = 0.3)')
    print('        -n, --num=<int>           Number of trees per class')
    print('                                  Available values: [1 .. INT_MAX] (default = 50)')
    print('        -r, --relax=<int>         Relax decimal points precision to this number')
    print('                                  Available values: [0 .. INT_MAX] (default = 0)')
    print('        -v, --verbose             Be verbose')

import signal

# Fonction qui sera appelée si le timeout est dépassé
def handler(signum, frame):
    raise TimeoutError("Timeout dépassé")


#
#==============================================================================
if __name__ == '__main__':
    depth, num, count, relax, verbose, files = parse_options()

    if files:
        datasets = files[0]
    else:
        datasets = 'datasets.list'

    with open(datasets, 'r') as fp:
        datasets = [line.strip() for line in fp.readlines() if line]

    print(f'training parameters: {num} trees per class, each of depth {"adaptive" if depth == -1 else depth}\n')

    # deleting the previous results
    if os.path.isdir('results'):
        shutil.rmtree('results')
    os.makedirs('results/smt')
    os.makedirs('results/mx')
    # os.makedirs('results/anchor')

    # initializing the seed
    #random.seed(1234)

    soptions = Options(f'./xreason.py --relax {relax} -z  -X abd -R lin -u -N 1 -e smt -x \'inst\' somefile'.split())
    moptions = Options(f'./xreason.py --relax {relax} -s g3 -z -X abd -R lin -u -N 1 -e mx -x \'inst\' somefile'.split())
    # ton dictionnaire des paramètres
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

    # training all XGBoost models
    for data in datasets:
        if depth != -1:
            adepth = depth
        else:
            # adaptive length
            data, adepth = data.split()

        print(f'processing {data}...')

        # reading and shuffling the instances
        with open(os.path.join(data), 'r') as fp:
            insts = [line.strip().rsplit(',', 1)[0] for line in fp.readlines()[1:]]
            #insts = list(set(insts))
            #insts = insts
            insts = list(dict.fromkeys(insts))
            #random.shuffle(insts)

            if count > 1:
                nof_insts = min(int(count), len(insts))
            else:
                nof_insts = min(int(len(insts) * count), len(insts))
            print(f'considering {nof_insts} instances')
        
        # récupérer les paramètres pour ce dataset
        # récupérer le nom de base sans extension pour le dossier et le fichier
        base = os.path.splitext(os.path.basename(data))[0]  # "arrowhead_0_vs_1"
        adepth, anum = param_dict.get(os.path.basename(data), (depth, num))

        # construire mfile avec le dossier et le fichier corrects
        mfile = f'temp/{base}/{base}_nbestim_{anum}_maxdepth_{adepth}_testsplit_0.3.mod.pkl'


        slog = open(f'results/smt/{base}.log', 'w')
        mlog = open(f'results/mx/{base}.log', 'w')

        # creating booster objects
        sxgb = XGBooster(soptions, from_model=mfile)
        sxgb.encode(test_on=None)
        mxgb = XGBooster(moptions, from_model=mfile)
        mxgb.encode(test_on=None)

        stimes = []
        mtimes = []
        mcalls = []
        smem = []
        mxmem = []


        #with open("/tmp/texture.samples", 'r') as fp:
        #    insts = [line.strip() for line in fp.readlines()]
        # Mettre le handler
        signal.signal(signal.SIGALRM, handler)
        # loop
        smt_timeouts = 0
        maxsat_timeouts = 0

        for i, inst in enumerate(insts):
            if i == nof_insts:
                break

            # préparer les données
            soptions.explain = [float(v.strip()) for v in inst.split(',')]
            moptions.explain = [float(v.strip()) for v in inst.split(',')]

            smt_success = False
            
            # ----- SMT -----
            try:
                signal.alarm(3*60)  # timeout SMT
                expl1 = sxgb.explain(soptions.explain)
                smem.append(round(sxgb.x.used_mem / 1024.0, 3))
                stimes.append(sxgb.x.time)

                # logging SMT
                print(f'i: {inst}', file=slog)
                print(f's: {len(expl1)}', file=slog)
                print(f't: {sxgb.x.time:.3f}', file=slog)
                print('', file=slog)

                smt_success = True
                
            except TimeoutError:
                smt_timeouts += 1
                print(f"[TIMEOUT SMT] instance {i}", file=slog)
            finally:
                signal.alarm(0)  # Toujours annuler l'alarme

            # ----- MaxSAT ----- (traiter même si SMT a échoué)
            try:
                signal.alarm(3*60)  # timeout MaxSAT
                expl2 = mxgb.explain(moptions.explain)
                mxmem.append(round(mxgb.x.used_mem / 1024.0, 3))
                mtimes.append(mxgb.x.time)
                mcalls.append(mxgb.x.calls)
                mxgb.x.calls = 0

                # logging MaxSAT
                print(f'i: {inst}', file=mlog)
                print(f's: {len(expl2[0])}', file=mlog)
                print(f't: {mxgb.x.time:.3f}', file=mlog)
                print(f'c: {mxgb.x.calls}', file=mlog)
                print('', file=mlog)

            except TimeoutError:
                maxsat_timeouts += 1
                print(f"[TIMEOUT MaxSAT] instance {i}", file=mlog)
            finally:
                signal.alarm(0)  # Toujours annuler l'alarme

            slog.flush()
            mlog.flush()
            sys.stdout.flush()

        # ----- après la boucle : affichage mémoire et stats -----
        # Affichage des statistiques SMT
        if stimes:
            print(f"SMT max time: {max(stimes):.2f}", file=slog)
            print(f"SMT min time: {min(stimes):.2f}", file=slog)
            print(f"SMT avg time: {sum(stimes)/len(stimes):.2f}", file=slog)
            print(f"SMT cumulative time: {sum(stimes):.2f}", file=slog)
            print(f"nombre of explain instances: {len(stimes)}", file=slog)
            print("all times:", ", ".join(f"{t:.3f}" for t in stimes), file=slog)
            if len(stimes) > 1:  # statistics.stdev nécessite au moins 2 valeurs
                print(f"SMT std dev: {statistics.stdev(stimes):.2f}", file=slog)
            print('', file=slog)

        # Affichage des statistiques MaxSAT
        if mtimes:
            print(f"MaxSAT max time: {max(mtimes):.2f}", file=mlog)
            print(f"MaxSAT min time: {min(mtimes):.2f}", file=mlog)
            print(f"MaxSAT avg time: {sum(mtimes)/len(mtimes):.2f}", file=mlog)
            print(f"MaxSAT cumulative time: {sum(mtimes):.2f}", file=mlog)
            print(f"nombre of explain instances: {len(mtimes)}", file=mlog)
            print("all times:", ", ".join(f"{t:.3f}" for t in mtimes), file=mlog)


            if len(mtimes) > 1:
                print(f"MaxSAT std dev: {statistics.stdev(mtimes):.2f}", file=mlog)
            print('', file=mlog)

        # Correction pour éviter les erreurs si les listes sont vides
        if smem and mxmem:
            print(f"mem usage: SMT={smem[-1]} MB MaxSAT={mxmem[-1]} MB", file=mlog)
        elif smem:
            print(f"mem usage: SMT={smem[-1]} MB MaxSAT=N/A MB", file=mlog)
        elif mxmem:
            print(f"mem usage: SMT=N/A MB MaxSAT={mxmem[-1]} MB", file=mlog)

        # ----- log des timeouts -----
        print(f"Nombre of timeouts SMT: {smt_timeouts}", file=slog)
        print(f"Nombre of timeouts MaxSAT: {maxsat_timeouts}", file=mlog)

        slog.flush()
        mlog.flush()

