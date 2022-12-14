# TODO move this but how ? New CLI args for convert ?
def init_parser(subparser):
    descr = "Generate a tileset from a set of geometries"
    parser = subparser.add_parser('export', help=descr)

    group = parser.add_mutually_exclusive_group()

    d_help = "Name of the directory containing the geometries"
    group.add_argument('-d', metavar='DIRECTORY', type=str, help=d_help)

    o_help = "Offset of the geometries (only with '-d')"
    parser.add_argument('-o', nargs=3, metavar=('X', 'Y', 'Z'), type=float, help=o_help)

    D_help = """
    Database connexion info (e.g. 'service=py3dtiles' or \
    'dbname=py3dtiles host=localhost port=5432 user=yourname password=yourpwd')
    """
    group.add_argument('-D', metavar='DB_CONNINFO', type=str, help=D_help)

    t_help = "Table name (required if '-D' option is activated)"
    parser.add_argument('-t', metavar='TABLE', type=str, help=t_help)

    c_help = "Geometry column name (required if '-D' option is activated)"
    parser.add_argument('-c', metavar='COLUMN', type=str, help=c_help)

    i_help = "Id column name (only with '-D')"
    parser.add_argument('-i', metavar='IDCOLUMN', type=str, help=i_help)

    return parser
