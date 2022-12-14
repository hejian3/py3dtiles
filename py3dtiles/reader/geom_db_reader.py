import getpass

import numpy as np
import psycopg2

def build_secure_conn(db_conn_info):
    """Get a psycopg2 connexion securely, e.g. without writing the password explicitely
    in the terminal

    Parameters
    ----------
    db_conn_info : str

    Returns
    -------
    psycopg2.extensions.connection
    """
    try:
        connection = psycopg2.connect(db_conn_info)
    except psycopg2.OperationalError:
        pw = getpass.getpass("Postgres password: ")
        connection = psycopg2.connect(db_conn_info + f" password={pw}")
    return connection

def from_db(db_conn_info, table_name, column_name, id_column_name):
    connection = build_secure_conn(db_conn_info)
    cur = connection.cursor()

    print("Loading data from database...")
    cur.execute(f"SELECT ST_3DExtent({column_name}) FROM {table_name}")
    extent = cur.fetchall()[0][0]
    extent = [m.split(" ") for m in extent[6:-1].split(",")]
    offset = [(float(extent[1][0]) + float(extent[0][0])) / 2,
              (float(extent[1][1]) + float(extent[0][1])) / 2,
              (float(extent[1][2]) + float(extent[0][2])) / 2]

    id_statement = ""
    if id_column_name is not None:
        id_statement = "," + id_column_name
    cur.execute("SELECT ST_AsBinary(ST_RotateX(ST_Translate({0}, {1}, {2}, {3}), -pi() / 2)),"
                "ST_Area(ST_Force2D({0})) AS weight{5} FROM {4} ORDER BY weight DESC"
                .format(column_name, -offset[0], -offset[1], -offset[2],
                        table_name, id_statement))
    res = cur.fetchall()
    [t[0] for t in res]
    if id_column_name is not None:
        [t[2] for t in res]
    transform = np.array([
        [1, 0, 0, offset[0]],
        [0, 1, 0, offset[1]],
        [0, 0, 1, offset[2]],
        [0, 0, 0, 1]], dtype=float)
    transform = transform.flatten('F')

    # wkbs_to_tileset(wkbs, ids, transform)
