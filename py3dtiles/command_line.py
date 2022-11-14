import argparse

from py3dtiles import convert, export, info, merger


def main():
    parser = argparse.ArgumentParser(
        description='Read/write 3dtiles files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_parsers = parser.add_subparsers(dest='command')

    # init subparsers
    command_parsers = [
        convert.init_parser(sub_parsers),
        info.init_parser(sub_parsers),
        merger.init_parser(sub_parsers),
        export.init_parser(sub_parsers)
    ]
    # add the verbose argument for all sub-parsers so that it is after the command.
    for command_parser in command_parsers:
        command_parser.add_argument('--verbose', '-v', action='count', default=0)

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
