import argparse

import py3dtiles.convert as convert
import py3dtiles.export as export
import py3dtiles.info as info
import py3dtiles.merger as merger


def main():
    parser = argparse.ArgumentParser(
        description='Read/write 3dtiles files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--verbose',
        help='Print logs (-1: no logs, 0: progress indicator, 1+: increased verbosity)',
        default=0, type=int)
    sub_parsers = parser.add_subparsers(dest='command')

    # init subparsers
    convert.init_parser(sub_parsers)
    info.init_parser(sub_parsers)
    merger.init_parser(sub_parsers)
    export.init_parser(sub_parsers)

    args = parser.parse_args()

    if args.command == 'convert':
        convert.main(args)
    elif args.command == 'info':
        info.main(args)
    elif args.command == 'merge':
        merger.main(args)
    elif args.command == 'export':
        export.main(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
