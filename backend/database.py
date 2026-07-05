import mysql.connector
from mysql.connector import pooling

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "",
    "database": "ai_triage_db"
}

pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="triage_pool",
    pool_size=10,
    **DB_CONFIG
)

def get_connection():
    return pool.get_connection()

def fetch_one(query, params=()):
    conn = cur = None
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        cur.execute(query, params)
        return cur.fetchone()
    finally:
        if cur:  cur.close()
        if conn: conn.close()  # always returns connection to pool

def fetch_all(query, params=()):
    conn = cur = None
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        cur.execute(query, params)
        return cur.fetchall()
    finally:
        if cur:  cur.close()
        if conn: conn.close()

def execute(query, params=()):
    conn = cur = None
    try:
        conn    = get_connection()
        cur     = conn.cursor()
        cur.execute(query, params)
        last_id = cur.lastrowid
        conn.commit()
        return last_id
    finally:
        if cur:  cur.close()
        if conn: conn.close()