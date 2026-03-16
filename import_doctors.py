"""
import_doctors.py
─────────────────
Run this ONCE from the project root to load all 1000 doctors
into the MySQL database.

Usage:
    cd Ai-triage-system
    python import_doctors.py
"""

import csv
import os
import sys
import mysql.connector

# ── Database config (same as database.py) ─────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "",           # ← change if you have a password
    "database": "ai_triage_db"
}

# ── Dataset path ──────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "dataset", "doctors_dataset_large.csv")

if not os.path.exists(CSV_PATH):
    print(f"❌ Dataset not found at: {CSV_PATH}")
    print("   Copy doctors_dataset_large.csv into the dataset/ folder and retry.")
    sys.exit(1)

def make_credentials(doctor_id: str, doctor_name: str):
    """
    username : firstname.DXXXX   e.g.  sanjay.D0001
    password : firstname + last4  e.g.  sanjay0001
    """
    # Extract first name (skip "Dr ")
    parts     = doctor_name.replace("Dr ", "").replace("Dr. ", "").strip().split()
    firstname = parts[0].lower() if parts else "doctor"
    last4     = doctor_id[-4:]          # "0001"
    username  = f"{firstname}.{doctor_id.lower()}"   # sanjay.d0001
    password  = f"{firstname}{last4}"                 # sanjay0001
    return username, password

def run_import():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Count existing
    cur.execute("SELECT COUNT(*) FROM doctors")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"⚠️  Doctors table already has {existing} rows.")
        ans = input("   Clear and re-import? (y/n): ").strip().lower()
        if ans == 'y':
            cur.execute("DELETE FROM doctors")
            conn.commit()
            print("   Cleared existing rows.")
        else:
            print("   Skipped. Exiting.")
            cur.close(); conn.close()
            return

    insert_sql = """
        INSERT INTO doctors
        (doctor_id, name, username, password, specialization,
         doctor_type, department, risk_level_handled,
         shift_start, shift_end, max_patients_per_hour,
         room_number, availability)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name),
            availability=VALUES(availability)
    """

    inserted = 0
    errors   = 0

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                did      = row['doctor_id'].strip()
                name     = row['doctor_name'].strip()
                username, password = make_credentials(did, name)

                cur.execute(insert_sql, (
                    did,
                    name,
                    username,
                    password,
                    row['specialization'].strip(),
                    row['doctor_type'].strip(),
                    row['department'].strip(),
                    row['risk_level_handled'].strip(),
                    row['shift_start'].strip(),
                    row['shift_end'].strip(),
                    int(row['max_patients_per_hour']),
                    int(row['room_number']),
                    row['availability'].strip()
                ))
                inserted += 1
                if inserted % 100 == 0:
                    print(f"   Inserted {inserted} doctors…")
            except Exception as e:
                errors += 1
                print(f"   ⚠️  Row error ({row.get('doctor_id','?')}): {e}")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n✅ Import complete!")
    print(f"   Inserted : {inserted}")
    print(f"   Errors   : {errors}")
    print(f"\n📋 Sample credentials:")
    print(f"   Dr Sanjay Agarwal  →  username: sanjay.d0001  password: sanjay0001")
    print(f"   Dr Anil Singh      →  username: anil.d0002    password: anil0002")
    print(f"\n   Pattern: firstname.DXXXX / firstnameLAST4DIGITS")

if __name__ == "__main__":
    run_import()