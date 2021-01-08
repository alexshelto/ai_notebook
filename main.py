

import argparse
from typing import Optional, Sequence
from Classifier.classifier import Classifier



def main(argv: Optional[Sequence[str]] = None) -> int:
    '''Driver function for classifier'''
    
    # Command line arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument('training', help='training data path')
    args = parser.parse_args(argv)
    print(args)
   

    network = Classifier(args.training) 
    return 0
    


if __name__ == '__main__':
    exit(main())
