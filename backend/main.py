import pickle, numpy as np, os, sys, random, string, shutil
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import fetch_one, fetch_all, execute
import uvicorn

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_pkl(name):
    path = os.path.join(BASE, "model", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "rb") as f: return pickle.load(f)

MODEL        = load_pkl("model.pkl")
LABEL_MAP    = load_pkl("label_map.pkl")
FEATURES     = load_pkl("features.pkl")
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

ACTIONS = {
    "GREEN":  "Patient is SAFE. All vitals are normal. No immediate attention required.",
    "YELLOW": "Consult general doctor within 2 hours.",
    "ORANGE": "Urgent — specialist consultation required today.",
    "RED":    "CRITICAL — Immediate emergency intervention required!",
}

HIGH_RISK_LEVELS   = {"RED", "ORANGE"}
NORMAL_RISK_LEVELS = {"YELLOW", "GREEN"}

SYMPTOM_SPEC_MAP = {
    "heart_disease_high":       ["Cardiology", "Cardiac Surgery", "Emergency Medicine"],
    "heart_disease_medium":     ["Cardiology", "Internal Medicine", "General Medicine"],
    "chest_pain_high":          ["Cardiology", "Emergency Medicine", "Internal Medicine"],
    "chest_pain_medium":        ["Cardiology", "General Medicine"],
    "breath_low_oxygen_high":   ["Pulmonology", "Emergency Medicine", "Respiratory Medicine"],
    "breath_low_oxygen_medium": ["Pulmonology", "Internal Medicine", "General Medicine"],
    "diabetes_high":            ["Endocrinology", "Internal Medicine", "Emergency Medicine"],
    "diabetes_medium":          ["Endocrinology", "General Medicine"],
    "fever_high":               ["Infectious Disease", "Emergency Medicine", "Internal Medicine"],
    "fever_medium":             ["General Medicine", "Internal Medicine"],
    "high_bp":                  ["Cardiology", "Nephrology", "Internal Medicine"],
    "RED":    ["Emergency Medicine", "Cardiology", "Internal Medicine"],
    "ORANGE": ["Internal Medicine", "Emergency Medicine", "Cardiology"],
    "YELLOW": ["General Medicine", "Internal Medicine"],
    "GREEN":  ["General Medicine", "General Practice"],
}

NORMAL_DOCTOR_SPECS = ["General Medicine", "General Practice", "Internal Medicine", "Family Medicine"]

app = FastAPI(title="AI Triage System", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND = os.path.join(BASE, "frontend")
if os.path.exists(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

class PatientIn(BaseModel):
    name: str; age: int; gender: str; phone: str
    address: Optional[str] = ""
    blood_group: Optional[str] = ""
    emergency_contact: Optional[str] = ""
    allergies: Optional[str] = ""
    existing_conditions: Optional[str] = ""
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

class VitalsIn(BaseModel):
    patient_id: int; heart_rate: int; oxygen: int; temperature: float
    systolic_bp: int; diastolic_bp: Optional[int] = 80
    respiratory_rate: int
    blood_glucose: Optional[float] = None
    weight_kg: Optional[float] = None; height_cm: Optional[float] = None
    consciousness: Optional[str] = "Alert"
    chest_pain: int = 0; breath_problem: int = 0; fever: int = 0
    diabetes: int = 0; heart_disease: int = 0; trauma: int = 0
    chief_complaint: Optional[str] = ""

class DoctorNoteIn(BaseModel):
    patient_id: int; recommended_tests: str
    diagnosis: Optional[str] = ""; appointment_time: Optional[str] = ""
    prescription: Optional[str] = ""

class MessageIn(BaseModel):
    patient_id: int; doctor_id: int; subject: str; body: str
    message_type: Optional[str] = "instruction"

class SlotIn(BaseModel):
    patient_id: int; doctor_id: int; slot_date: str; slot_time: str
    notes: Optional[str] = ""

class ReportIn(BaseModel):
    patient_id: int; report_text: str

class DoctorLogin(BaseModel):
    username: str; password: str

class PatientLookup(BaseModel):
    name: Optional[str] = ""; phone: Optional[str] = ""; uid: Optional[str] = ""

class StatusUpdate(BaseModel):
    patient_id: int; status: str

def generate_uid():
    uid = "PT-" + "".join(random.choices(string.ascii_uppercase, k=3)) + \
          "".join(random.choices(string.digits, k=4))
    while fetch_one("SELECT id FROM patients WHERE uid=%s", (uid,)):
        uid = generate_uid()
    return uid

def build_features(v: VitalsIn, age: int):
    hr, o2, bp, rr, tmp = v.heart_rate, v.oxygen, v.systolic_bp, v.respiratory_rate, v.temperature
    si = hr/bp; hor = hr/o2; ror = rr/o2; thr = tmp/hr
    ar = int(age>60); lo = int(o2<92); vlo = int(o2<85)
    hhr = int(hr>100); vhhr = int(hr>130); lbp = int(bp<90)
    ht = int(tmp>101); hresp = int(rr>20); vhresp = int(rr>30)
    gr = int((v.blood_glucose or 0)>200 or (v.blood_glucose or 100)<70)
    cr = int(v.consciousness not in ("Alert","alert"))
    ds = (ar + lo*2 + vlo*3 + hhr + vhhr*2 + lbp*2 + ht + hresp + vhresp*2 +
          v.chest_pain*2 + v.breath_problem*2 + v.fever + v.diabetes + v.heart_disease*2 +
          gr*2 + cr*3)
    return np.array([[age,hr,o2,tmp,bp,rr,v.chest_pain,v.breath_problem,v.fever,v.diabetes,v.heart_disease,
                      si,hor,ror,thr,ar,lo,vlo,hhr,vhhr,lbp,ht,hresp,vhresp,ds,
                      age*hr/1000,bp*o2/10000,hr*rr/1000,o2*bp/10000]])

def assign_doctor_preview(specs: list) -> Optional[dict]:
    for spec in specs:
        candidates = fetch_all("""SELECT * FROM doctors
            WHERE (specialization=%s OR department LIKE %s) AND availability='Available'
            ORDER BY current_patients ASC LIMIT 1""", (spec, f"%{spec}%"))
        if candidates: return candidates[0]
    return fetch_one("SELECT * FROM doctors WHERE availability='Available' ORDER BY current_patients ASC LIMIT 1")

def detect_specialization(v: VitalsIn, severity: str, age: int) -> list:
    level = "high" if severity in HIGH_RISK_LEVELS else "medium"
    specs = []
    if v.heart_disease:  specs += SYMPTOM_SPEC_MAP[f"heart_disease_{level}"]
    if v.chest_pain:     specs += SYMPTOM_SPEC_MAP[f"chest_pain_{level}"]
    if v.breath_problem or v.oxygen < 92: specs += SYMPTOM_SPEC_MAP[f"breath_low_oxygen_{level}"]
    if v.diabetes:       specs += SYMPTOM_SPEC_MAP[f"diabetes_{level}"]
    if v.fever or v.temperature > 101:    specs += SYMPTOM_SPEC_MAP[f"fever_{level}"]
    if v.systolic_bp > 160:               specs += SYMPTOM_SPEC_MAP["high_bp"]
    if age > 65 and severity in HIGH_RISK_LEVELS:
        specs = ["Emergency Medicine","Geriatrics"] + specs
    if not specs: specs = SYMPTOM_SPEC_MAP.get(severity, ["General Medicine"])
    seen, result = set(), []
    for s in specs:
        if s not in seen: seen.add(s); result.append(s)
    return result

def assign_doctor(severity: str, patient_id: int, vitals: VitalsIn = None, age: int = 30) -> Optional[dict]:
    is_high_risk = severity in HIGH_RISK_LEVELS
    if is_high_risk and vitals: preferred_specs = detect_specialization(vitals, severity, age)
    elif is_high_risk:          preferred_specs = SYMPTOM_SPEC_MAP.get(severity, ["Emergency Medicine"])
    else:                       preferred_specs = NORMAL_DOCTOR_SPECS

    for spec in preferred_specs:
        candidates = fetch_all("""SELECT * FROM doctors WHERE (specialization=%s OR department LIKE %s)
            AND availability='Available' ORDER BY current_patients ASC LIMIT 5""", (spec, f"%{spec}%"))
        if candidates:
            doc = candidates[0]; _finalize_assignment(doc["id"], patient_id); return doc

    for spec in preferred_specs[:2]:
        candidates = fetch_all("""SELECT * FROM doctors WHERE (specialization=%s OR department LIKE %s)
            ORDER BY current_patients ASC LIMIT 3""", (spec, f"%{spec}%"))
        if candidates:
            doc = candidates[0]; _finalize_assignment(doc["id"], patient_id); return doc

    any_doc = fetch_one("SELECT * FROM doctors WHERE availability='Available' ORDER BY current_patients ASC LIMIT 1")
    if any_doc: _finalize_assignment(any_doc["id"], patient_id); return any_doc
    last = fetch_one("SELECT * FROM doctors ORDER BY current_patients ASC LIMIT 1")
    if last: _finalize_assignment(last["id"], patient_id); return last
    return None

def _finalize_assignment(doctor_db_id: int, patient_id: int):
    execute("UPDATE doctors SET current_patients=current_patients+1 WHERE id=%s", (doctor_db_id,))
    execute("UPDATE patients SET assigned_doctor_id=%s WHERE id=%s", (doctor_db_id, patient_id))

def release_doctor_slot(patient_id: int):
    p = fetch_one("SELECT assigned_doctor_id FROM patients WHERE id=%s", (patient_id,))
    if p and p.get("assigned_doctor_id"):
        execute("UPDATE doctors SET current_patients=GREATEST(0,current_patients-1) WHERE id=%s",
                (p["assigned_doctor_id"],))

@app.get("/")
def root(): return {"status": "AI Triage System v5.0.0 OK"}

@app.get("/health")
def health(): return {"status": "ok", "model": type(MODEL).__name__}

@app.post("/api/doctor/login")
def doctor_login(data: DoctorLogin):
    doc = fetch_one("SELECT * FROM doctors WHERE username=%s AND password=%s",
                    (data.username.lower(), data.password))
    if not doc: raise HTTPException(401, "Invalid username or password")
    return {"success": True, "doctor_db_id": doc["id"], "doctor_id": doc["doctor_id"],
            "name": doc["name"], "speciality": doc["specialization"], "department": doc["department"],
            "room": doc["room_number"], "availability": doc["availability"],
            "username": data.username, "message": f"Welcome, {doc['name']}!"}

@app.get("/api/doctor/profile/{doctor_db_id}")
def doctor_profile(doctor_db_id: int):
    doc = fetch_one("SELECT * FROM doctors WHERE id=%s", (doctor_db_id,))
    if not doc: raise HTTPException(404, "Doctor not found")
    return doc

@app.post("/api/patient/register")
def register(data: PatientIn):
    ex = fetch_one("SELECT id,uid,name FROM patients WHERE phone=%s", (data.phone,))
    if ex: raise HTTPException(409, f"Phone already registered. UID: {ex['uid']}, Name: {ex['name']}")
    uid = generate_uid()
    bmi = round(data.weight_kg/((data.height_cm/100)**2), 1) if data.weight_kg and data.height_cm else None
    pid = execute("""INSERT INTO patients
        (name,age,gender,phone,address,blood_group,emergency_contact,
         allergies,existing_conditions,weight_kg,height_cm,bmi,uid)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (data.name,data.age,data.gender,data.phone,data.address,data.blood_group,
         data.emergency_contact,data.allergies,data.existing_conditions,
         data.weight_kg,data.height_cm,bmi,uid))
    return {"success": True, "patient_id": pid, "uid": uid, "message": f"Patient {data.name} registered."}

@app.post("/api/patient/lookup")
def lookup(data: PatientLookup):
    p = None
    if data.uid:     p = fetch_one("SELECT * FROM patients WHERE uid=%s", (data.uid.upper(),))
    elif data.phone: p = fetch_one("SELECT * FROM patients WHERE phone=%s", (data.phone,))
    elif data.name:  p = fetch_one("SELECT * FROM patients WHERE name LIKE %s LIMIT 1", (f"%{data.name}%",))
    if not p: raise HTTPException(404, "Patient not found")
    safe = {k: v for k, v in p.items() if k not in ("id","assigned_doctor_id")}
    v    = fetch_one("SELECT severity FROM vitals WHERE patient_id=%s ORDER BY id DESC LIMIT 1", (p["id"],))
    return {"found": True, "patient": safe, "internal_id": p["id"],
            "has_vitals": bool(v), "severity": (v or {}).get("severity")}

@app.get("/api/nurse/patients")
def nurse_patients():
    rows = fetch_all("""SELECT p.uid,p.name,p.age,p.gender,p.phone,p.blood_group,p.photo_url,
               p.status,p.registered_at,p.existing_conditions,p.allergies,
               v.severity,v.risk_score,v.recorded_at as last_triage
        FROM patients p
        LEFT JOIN vitals v ON v.id=(SELECT MAX(id) FROM vitals WHERE patient_id=p.id)
        ORDER BY p.registered_at DESC""")
    return {"patients": rows, "total": len(rows)}

@app.get("/api/nurse/internal-id/{uid}")
def get_internal_id(uid: str):
    p = fetch_one("""SELECT id,name,age,gender,phone,blood_group,allergies,
               existing_conditions,weight_kg,height_cm FROM patients WHERE uid=%s""", (uid.upper(),))
    if not p: raise HTTPException(404, "Not found")
    return p

@app.post("/api/nurse/vitals")
def vitals(v: VitalsIn):
    patient = fetch_one("SELECT * FROM patients WHERE id=%s", (v.patient_id,))
    if not patient: raise HTTPException(404, "Patient not found")

    features   = build_features(v, patient["age"])
    pred       = MODEL.predict(features)[0]
    proba      = MODEL.predict_proba(features)[0]
    severity   = IDX_TO_LABEL[int(pred)]
    risk_score = round(float(proba[int(pred)]) * 100, 1)
    action     = ACTIONS[severity]
    prob_dict  = {IDX_TO_LABEL[i]: round(float(p)*100, 1) for i, p in enumerate(proba)}

    if v.trauma and severity == "GREEN":
        severity   = "YELLOW"
        risk_score = max(risk_score, 45.0)
        action     = "Trauma patient — doctor review required even with normal vitals."
        print(f"[TRAUMA] Patient #{v.patient_id} — GREEN overridden to YELLOW.")

    if severity == "GREEN":
        green_doc = assign_doctor_preview(NORMAL_DOCTOR_SPECS)
        print(f"[GREEN] Patient #{v.patient_id} is SAFE — not stored in DB.")
        return {
            "success": True, "severity": "GREEN", "risk_score": 0,
            "action": "Patient is SAFE. All vitals are normal. Not stored in database.",
            "probabilities": prob_dict, "is_high_risk": False, "is_critical": False,
            "assigned_doctor": {
                "name":           green_doc["name"]           if green_doc else None,
                "specialization": green_doc["specialization"] if green_doc else None,
                "room":           green_doc["room_number"]    if green_doc else None,
                "doctor_id":      green_doc["doctor_id"]      if green_doc else None,
                "availability":   green_doc["availability"]   if green_doc else None,
                "username":       green_doc["username"]       if green_doc else None,
                "password":       green_doc["password"]       if green_doc else None,
            } if green_doc else None
        }

    ai_summary = f"{severity}|{risk_score}|{action}"
    bmi = round(v.weight_kg/((v.height_cm/100)**2), 1) if v.weight_kg and v.height_cm else None

    execute("""INSERT INTO vitals
        (patient_id,heart_rate,oxygen,temperature,systolic_bp,diastolic_bp,
         respiratory_rate,blood_glucose,weight_kg,height_cm,bmi,consciousness,
         chest_pain,breath_problem,fever,diabetes,heart_disease,
         severity,risk_score,ai_summary,chief_complaint)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (v.patient_id,v.heart_rate,v.oxygen,v.temperature,v.systolic_bp,v.diastolic_bp,
         v.respiratory_rate,v.blood_glucose,v.weight_kg,v.height_cm,bmi,v.consciousness,
         v.chest_pain,v.breath_problem,v.fever,v.diabetes,v.heart_disease,
         severity,risk_score,ai_summary,v.chief_complaint))

    execute("UPDATE patients SET status=%s WHERE id=%s", (severity.lower(), v.patient_id))

    is_high_risk = severity in HIGH_RISK_LEVELS
    assigned_doc = assign_doctor(severity, v.patient_id, vitals=v, age=patient["age"])

    if assigned_doc:
        msg_body = (f"You have been triaged as {severity} with risk score {risk_score}%. "
                    f"Referred to: {assigned_doc['name']} ({assigned_doc['specialization']}). "
                    f"Please wait for appointment confirmation.") if is_high_risk else \
                   (f"Triage complete. Severity: {severity}. "
                    f"A doctor has been assigned and will send instructions shortly.")
        msg_type = "specialist_referral" if is_high_risk else "triage_complete"
        execute("INSERT INTO messages (patient_id,doctor_id,message_type,subject,body) VALUES(%s,%s,%s,%s,%s)",
                (v.patient_id,assigned_doc["id"],msg_type,"Triage Complete — Your Assessment Result",msg_body))

    return {
        "success": True, "severity": severity, "risk_score": risk_score, "action": action,
        "probabilities": prob_dict, "is_high_risk": is_high_risk,
        "is_critical": severity in ("RED","ORANGE"),
        "assigned_doctor": {
            "name":           assigned_doc["name"]           if assigned_doc else None,
            "specialization": assigned_doc["specialization"] if assigned_doc else None,
            "room":           assigned_doc["room_number"]    if assigned_doc else None,
            "doctor_id":      assigned_doc["doctor_id"]      if assigned_doc else None,
            "availability":   assigned_doc["availability"]   if assigned_doc else None,
            "username":       assigned_doc["username"]       if assigned_doc else None,
            "password":       assigned_doc["password"]       if assigned_doc else None,
        } if assigned_doc else None
    }

@app.get("/api/doctor/patients")
def doctor_patients(doctor_db_id: Optional[int] = None):
    if doctor_db_id:
        rows = fetch_all("""SELECT p.id,p.name,p.age,p.gender,p.phone,p.uid,p.blood_group,p.photo_url,
                   p.address,p.allergies,p.existing_conditions,p.emergency_contact,
                   p.status,p.registered_at,v.severity,v.risk_score,v.ai_summary,
                   v.recorded_at as last_triage,v.heart_rate,v.oxygen,v.temperature,
                   v.systolic_bp,v.respiratory_rate,v.consciousness,v.chief_complaint,
                   v.chest_pain,v.breath_problem,v.fever,v.diabetes,v.heart_disease
            FROM patients p
            INNER JOIN vitals v ON v.id=(SELECT MAX(id) FROM vitals WHERE patient_id=p.id)
            WHERE p.assigned_doctor_id=%s AND v.risk_score > 0
            ORDER BY CASE p.status WHEN 'solved' THEN 99 ELSE 0 END,
                     CASE v.severity WHEN 'RED' THEN 1 WHEN 'ORANGE' THEN 2
                                     WHEN 'YELLOW' THEN 3 ELSE 4 END,
                     v.risk_score DESC""", (doctor_db_id,))
    else:
        rows = fetch_all("""SELECT p.id,p.name,p.age,p.gender,p.phone,p.uid,p.blood_group,p.photo_url,
                   p.address,p.allergies,p.existing_conditions,p.emergency_contact,
                   p.status,p.registered_at,v.severity,v.risk_score,v.ai_summary,
                   v.recorded_at as last_triage,v.heart_rate,v.oxygen,v.temperature,
                   v.systolic_bp,v.respiratory_rate,v.consciousness,v.chief_complaint,
                   v.chest_pain,v.breath_problem,v.fever,v.diabetes,v.heart_disease
            FROM patients p
            INNER JOIN vitals v ON v.id=(SELECT MAX(id) FROM vitals WHERE patient_id=p.id)
            WHERE v.risk_score > 0
            ORDER BY CASE p.status WHEN 'solved' THEN 99 ELSE 0 END,
                     CASE v.severity WHEN 'RED' THEN 1 WHEN 'ORANGE' THEN 2
                                     WHEN 'YELLOW' THEN 3 ELSE 4 END,
                     v.risk_score DESC""")
    result = {"critical":[],"urgent":[],"moderate":[],"routine":[],"solved":[],"total":len(rows)}
    for r in rows:
        if r.get("status")=="solved":     result["solved"].append(r)
        elif r.get("severity")=="RED":    result["critical"].append(r)
        elif r.get("severity")=="ORANGE": result["urgent"].append(r)
        elif r.get("severity")=="YELLOW": result["moderate"].append(r)
        else:                             result["routine"].append(r)
    return result

@app.get("/api/doctor/patient/{pid}")
def doctor_patient(pid: int):
    p  = fetch_one("SELECT * FROM patients WHERE id=%s", (pid,))
    if not p: raise HTTPException(404, "Not found")
    v  = fetch_one("SELECT * FROM vitals WHERE patient_id=%s ORDER BY id DESC LIMIT 1", (pid,))
    n  = fetch_one("SELECT * FROM doctor_notes WHERE patient_id=%s ORDER BY id DESC LIMIT 1", (pid,))
    rs = fetch_all("SELECT * FROM reports WHERE patient_id=%s ORDER BY uploaded_at DESC", (pid,))
    ds = fetch_all("SELECT * FROM patient_docs WHERE patient_id=%s ORDER BY uploaded_at DESC", (pid,))
    assigned_doc = fetch_one("SELECT * FROM doctors WHERE id=%s", (p["assigned_doctor_id"],)) \
                   if p.get("assigned_doctor_id") else None
    return {"patient": p, "vitals": v, "notes": n, "reports": rs, "docs": ds, "assigned_doctor": assigned_doc}

@app.post("/api/doctor/status")
def update_status(data: StatusUpdate):
    if data.status == "solved": release_doctor_slot(data.patient_id)
    execute("UPDATE patients SET status=%s WHERE id=%s", (data.status, data.patient_id))
    return {"success": True}

@app.post("/api/doctor/recommend")
def recommend(data: DoctorNoteIn):
    execute("""INSERT INTO doctor_notes (patient_id,recommended_tests,diagnosis,appointment_time,prescription)
               VALUES(%s,%s,%s,%s,%s)""",
            (data.patient_id,data.recommended_tests,data.diagnosis,data.appointment_time,data.prescription))
    execute("UPDATE patients SET status='under_review' WHERE id=%s", (data.patient_id,))
    return {"success": True, "message": "Recommendation saved."}

@app.post("/api/doctor/send-message")
def send_message(data: MessageIn):
    execute("INSERT INTO messages (patient_id,doctor_id,message_type,subject,body) VALUES(%s,%s,%s,%s,%s)",
            (data.patient_id,data.doctor_id,data.message_type,data.subject,data.body))
    execute("UPDATE patients SET status='under_review' WHERE id=%s", (data.patient_id,))
    return {"success": True, "message": "Message sent to patient."}

@app.post("/api/doctor/book-slot")
def book_slot(data: SlotIn):
    execute("DELETE FROM specialist_slots WHERE patient_id=%s AND doctor_id=%s",
            (data.patient_id,data.doctor_id))
    execute("INSERT INTO specialist_slots (doctor_id,patient_id,slot_date,slot_time,notes) VALUES(%s,%s,%s,%s,%s)",
            (data.doctor_id,data.patient_id,data.slot_date,data.slot_time,data.notes))
    doc = fetch_one("SELECT * FROM doctors WHERE id=%s", (data.doctor_id,))
    if doc:
        execute("INSERT INTO messages (patient_id,doctor_id,message_type,subject,body) VALUES(%s,%s,'appointment',%s,%s)",
                (data.patient_id,data.doctor_id,"Specialist Appointment Confirmed",
                 f"Your appointment with {doc['name']} ({doc['specialization']}) is confirmed for "
                 f"{data.slot_date} at {data.slot_time}. Room: {doc['room_number']}. "
                 f"{data.notes or 'Please arrive 10 minutes early.'}"))
    return {"success": True, "message": "Slot booked and patient notified."}

@app.get("/api/patient/messages/{pid}")
def patient_messages(pid: int):
    msgs = fetch_all("""SELECT m.*, d.name as doctor_name, d.specialization, d.room_number
                        FROM messages m LEFT JOIN doctors d ON m.doctor_id=d.id
                        WHERE m.patient_id=%s ORDER BY m.sent_at DESC""", (pid,))
    execute("UPDATE messages SET is_read=1 WHERE patient_id=%s AND is_read=0", (pid,))
    return {"messages": msgs, "total": len(msgs), "unread": sum(1 for m in msgs if not m.get("is_read"))}

@app.get("/api/patient/unread-count/{pid}")
def unread_count(pid: int):
    row = fetch_one("SELECT COUNT(*) AS n FROM messages WHERE patient_id=%s AND is_read=0", (pid,))
    return {"unread": row["n"] if row else 0}

@app.get("/api/patient/status/{pid}")
def patient_status(pid: int):
    patient = fetch_one("SELECT * FROM patients WHERE id=%s", (pid,))
    if not patient: raise HTTPException(404, "Patient not found")
    vitals = fetch_one("SELECT * FROM vitals WHERE patient_id=%s ORDER BY id DESC LIMIT 1", (pid,))
    notes  = fetch_one("SELECT * FROM doctor_notes WHERE patient_id=%s ORDER BY id DESC LIMIT 1", (pid,))
    msgs   = fetch_all("""SELECT m.*, d.name as doctor_name, d.specialization, d.room_number
                          FROM messages m LEFT JOIN doctors d ON m.doctor_id=d.id
                          WHERE m.patient_id=%s ORDER BY m.sent_at DESC""", (pid,))
    severity     = (vitals or {}).get("severity", "")
    is_high_risk = severity in HIGH_RISK_LEVELS
    specialist_info = slot_info = None
    if is_high_risk and patient.get("assigned_doctor_id"):
        doc = fetch_one("SELECT * FROM doctors WHERE id=%s", (patient["assigned_doctor_id"],))
        if doc:
            specialist_info = {"name": doc["name"], "specialization": doc["specialization"],
                               "department": doc["department"], "room_number": doc["room_number"],
                               "availability": doc["availability"], "doctor_id": doc["doctor_id"]}
            slot_info = fetch_one("""SELECT * FROM specialist_slots WHERE patient_id=%s AND doctor_id=%s
                                     ORDER BY created_at DESC LIMIT 1""", (pid,doc["id"]))
    execute("UPDATE messages SET is_read=1 WHERE patient_id=%s AND is_read=0", (pid,))
    return {"patient": patient, "vitals": vitals, "notes": notes, "messages": msgs,
            "is_high_risk": is_high_risk, "specialist_info": specialist_info, "slot_info": slot_info}

@app.post("/api/patient/upload-photo")
async def upload_photo(patient_id: int = Form(...), file: UploadFile = File(...)):
    ext = file.filename.rsplit(".",1)[-1].lower()
    if ext not in ("jpg","jpeg","png","webp"): raise HTTPException(400, "Only JPG/PNG allowed")
    fname = f"photo_{patient_id}_{random.randint(1000,9999)}.{ext}"
    with open(os.path.join(UPLOAD_DIR, fname), "wb") as f: shutil.copyfileobj(file.file, f)
    url = f"/uploads/{fname}"
    execute("UPDATE patients SET photo_url=%s WHERE id=%s", (url,patient_id))
    return {"success": True, "url": url}

@app.post("/api/patient/upload-doc")
async def upload_doc(patient_id: int = Form(...), doc_type: str = Form("report"), file: UploadFile = File(...)):
    ext = file.filename.rsplit(".",1)[-1].lower()
    if ext not in ("pdf","jpg","jpeg","png"): raise HTTPException(400, "Only PDF/JPG/PNG")
    fname = f"doc_{patient_id}_{random.randint(10000,99999)}.{ext}"
    with open(os.path.join(UPLOAD_DIR, fname), "wb") as f: shutil.copyfileobj(file.file, f)
    url = f"/uploads/{fname}"
    execute("INSERT INTO patient_docs(patient_id,doc_type,file_url,file_name,file_ext) VALUES(%s,%s,%s,%s,%s)",
            (patient_id,doc_type,url,file.filename,ext))
    return {"success": True, "url": url}

@app.get("/api/patient/docs/{pid}")
def patient_docs(pid: int):
    return {"docs": fetch_all("SELECT * FROM patient_docs WHERE patient_id=%s ORDER BY uploaded_at DESC", (pid,))}

@app.post("/api/patient/report")
def patient_report(data: ReportIn):
    execute("INSERT INTO reports (patient_id,report_text) VALUES(%s,%s)", (data.patient_id,data.report_text))
    execute("UPDATE patients SET status='report_uploaded' WHERE id=%s", (data.patient_id,))
    return {"success": True}

@app.get("/api/stats")
def stats():
    total     = fetch_one("SELECT COUNT(*) AS n FROM patients")["n"]
    critical  = fetch_one("SELECT COUNT(*) AS n FROM vitals WHERE severity='RED'")["n"]
    orange    = fetch_one("SELECT COUNT(*) AS n FROM vitals WHERE severity='ORANGE'")["n"]
    waiting   = fetch_one("SELECT COUNT(*) AS n FROM patients WHERE status='waiting'")["n"]
    solved    = fetch_one("SELECT COUNT(*) AS n FROM patients WHERE status='solved'")["n"]
    doc_avail = fetch_one("SELECT COUNT(*) AS n FROM doctors WHERE availability='Available'")["n"]
    return {"total": total, "critical": critical, "orange": orange,
            "waiting": waiting, "solved": solved, "doctors_available": doc_avail}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)