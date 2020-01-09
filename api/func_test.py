import argparse
parser = argparse.ArgumentParser(description="This is a predict program")

parser.add_argument('-d', help='Data location')
parser.add_argument('-m', help='Model')
parser.add_argument('-f', help='testing f')
 
args = parser.parse_args()

if args.d:
    print('Data  :', args.d)
if args.m:
    print('Model :', args.m)
if args.f:
    print('flag f :', args.f)